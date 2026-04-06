//! Build cargo_metadata::Metadata from Cargo.lock + Cargo.toml files
//! without invoking `cargo metadata`.
//!
//! This enables fully sandboxed `crate2nix generate` with no network access
//! and no cargo binary required.

use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};

use anyhow::{format_err, Context, Error};
use cargo_metadata::Metadata;
use serde_json::json;

/// A git source mapping: (url, rev, local_path)
#[derive(Debug, Clone)]
pub struct GitSourceMapping {
    /// The git URL (e.g., "https://github.com/user/repo")
    pub url: String,
    /// The git revision hash
    pub rev: String,
    /// Local path to the pre-fetched source
    pub local_path: PathBuf,
}

impl GitSourceMapping {
    /// Parse a "url#rev=path" string into a GitSourceMapping
    pub fn parse(s: &str) -> Result<GitSourceMapping, Error> {
        let last_eq = s
            .rfind('=')
            .ok_or_else(|| format_err!("Invalid git-source '{}': expected 'url#rev=path'", s))?;
        let (url_rev, local_path) = (&s[..last_eq], &s[last_eq + 1..]);
        let (url, rev) = url_rev
            .rsplit_once('#')
            .ok_or_else(|| format_err!("Invalid git-source '{}': expected 'url#rev=path'", s))?;
        Ok(GitSourceMapping {
            url: url.to_string(),
            rev: rev.to_string(),
            local_path: PathBuf::from(local_path),
        })
    }
}

/// Configuration for building metadata from lockfile
pub struct LockfileConfig {
    /// Path to the Cargo.lock file
    pub cargo_lock: PathBuf,
    /// Path to the workspace root (directory containing root Cargo.toml)
    pub workspace_root: PathBuf,
    /// Git source mappings
    pub git_sources: Vec<GitSourceMapping>,
    /// Optional path to directory with extracted crates.io manifests
    /// Structure: manifests_dir/name/version/Cargo.toml
    pub crates_io_manifests: Option<PathBuf>,
}

/// A package parsed from Cargo.lock
#[derive(Debug, Clone)]
struct LockPackage {
    name: String,
    version: String,
    source: Option<String>,
    #[allow(dead_code)]
    checksum: Option<String>,
    dependencies: Vec<String>,
}

/// Dependency specification parsed from Cargo.toml
#[derive(Debug, Clone)]
struct DepSpec {
    name: String,
    package: Option<String>,
    version_req: String,
    optional: bool,
    default_features: bool,
    features: Vec<String>,
    source: Option<String>,
}

/// Build `cargo_metadata::Metadata` from Cargo.lock + Cargo.toml files.
pub fn build_metadata(config: &LockfileConfig) -> Result<Metadata, Error> {
    let lock_content = std::fs::read_to_string(&config.cargo_lock).context("reading Cargo.lock")?;
    let lock_packages = parse_cargo_lock(&lock_content)?;

    // Parse workspace root Cargo.toml
    let root_toml_path = config.workspace_root.join("Cargo.toml");
    let root_toml_content =
        std::fs::read_to_string(&root_toml_path).context("reading root Cargo.toml")?;
    let root_toml: toml::Value = root_toml_content
        .parse()
        .context("parsing root Cargo.toml")?;

    // Get workspace members
    let workspace_members = get_workspace_members(&root_toml, &config.workspace_root)?;

    // Build git source index: source_string -> local_path
    let git_source_index = build_git_source_index(&config.git_sources, &lock_packages)?;

    // Build package ID for each lock package
    let mut packages_json = Vec::new();
    let mut nodes_json = Vec::new();
    let mut workspace_member_ids = Vec::new();

    // Create a lookup from (name, version, source) -> lock package index.
    // Also index by source without the #commit suffix, since Cargo.lock dep
    // strings reference sources without the commit hash.
    let mut lock_pkg_lookup: HashMap<(String, String, Option<String>), usize> = HashMap::new();
    for (i, pkg) in lock_packages.iter().enumerate() {
        lock_pkg_lookup.insert(
            (pkg.name.clone(), pkg.version.clone(), pkg.source.clone()),
            i,
        );
        // Also insert with source stripped of #commit for dep string matching
        if let Some(source) = &pkg.source {
            if let Some(hash_pos) = source.find('#') {
                let stripped = source[..hash_pos].to_string();
                lock_pkg_lookup.insert((pkg.name.clone(), pkg.version.clone(), Some(stripped)), i);
            }
        }
    }

    // Build package ID strings
    let pkg_ids: Vec<String> = lock_packages
        .iter()
        .map(|pkg| make_package_id(pkg))
        .collect();

    // Resolve workspace Cargo.toml data for workspace inheritance
    let workspace_package = root_toml.get("workspace").and_then(|ws| ws.get("package"));
    let workspace_deps = root_toml
        .get("workspace")
        .and_then(|ws| ws.get("dependencies"));

    for (i, lock_pkg) in lock_packages.iter().enumerate() {
        let pkg_id = &pkg_ids[i];

        // Determine if this is a workspace member
        let is_workspace_member = workspace_members
            .iter()
            .any(|(name, _)| name == &lock_pkg.name);
        let mut member_path = workspace_members
            .iter()
            .find(|(name, _)| name == &lock_pkg.name)
            .map(|(_, path)| path.clone());

        // For path deps (no source) that aren't workspace members,
        // search the workspace root for their Cargo.toml.
        // These are typically sub-crates of workspace members.
        if member_path.is_none() && lock_pkg.source.is_none() {
            if let Some(found) = find_manifest_recursive(&config.workspace_root, &lock_pkg.name, 5)
            {
                member_path = found.parent().map(|p| p.to_path_buf());
            }
        }

        // Try to find and read the Cargo.toml for this package
        let manifest_data = find_and_read_manifest(
            lock_pkg,
            &member_path,
            &config.workspace_root,
            &git_source_index,
            &config.crates_io_manifests,
            workspace_package,
            workspace_deps,
        );

        // Build the Package JSON
        let (pkg_json, dep_specs) = build_package_json(
            lock_pkg,
            pkg_id,
            &config.workspace_root,
            &member_path,
            &manifest_data,
        )?;
        packages_json.push(pkg_json);

        // Treat all local path deps (no source in Cargo.lock) as workspace members
        // for source resolution purposes. This includes sub-crates of workspace
        // members that aren't directly listed in [workspace.members].
        if is_workspace_member || lock_pkg.source.is_none() {
            workspace_member_ids.push(pkg_id.clone());
        }

        // Build the Node for the resolve graph
        let node_json = build_node_json(
            lock_pkg,
            pkg_id,
            &dep_specs,
            &lock_packages,
            &pkg_ids,
            &lock_pkg_lookup,
        )?;
        nodes_json.push(node_json);
    }

    // Determine root package
    let root_id = if workspace_member_ids.len() == 1 {
        Some(workspace_member_ids[0].clone())
    } else {
        None
    };

    let metadata_json = json!({
        "version": 1,
        "packages": packages_json,
        "workspace_members": workspace_member_ids,
        "workspace_root": config.workspace_root.to_string_lossy(),
        "target_directory": config.workspace_root.join("target").to_string_lossy().to_string(),
        "resolve": {
            "root": root_id,
            "nodes": nodes_json,
        },
    });

    // Write debug JSON to help diagnose deserialization errors
    if std::env::var("CRATE2NIX_DEBUG").is_ok() {
        let debug_path = std::env::var("CRATE2NIX_DEBUG")
            .unwrap_or_else(|_| "/tmp/debug-metadata.json".to_string());
        if let Ok(json_str) = serde_json::to_string_pretty(&metadata_json) {
            let _ = std::fs::write(&debug_path, &json_str);
            eprintln!("DEBUG: wrote metadata JSON to {}", debug_path);
        }
    }

    let metadata: Metadata =
        serde_json::from_value(metadata_json).context("deserializing constructed metadata")?;

    Ok(metadata)
}

/// Parse Cargo.lock TOML into a list of LockPackage
fn parse_cargo_lock(content: &str) -> Result<Vec<LockPackage>, Error> {
    let lock: toml::Value = content.parse().context("parsing Cargo.lock TOML")?;

    let packages = lock
        .get("package")
        .and_then(|v| v.as_array())
        .ok_or_else(|| format_err!("Cargo.lock missing [[package]] array"))?;

    let mut result = Vec::new();
    for pkg in packages {
        let name = pkg
            .get("name")
            .and_then(|v| v.as_str())
            .ok_or_else(|| format_err!("package missing name"))?
            .to_string();
        let version = pkg
            .get("version")
            .and_then(|v| v.as_str())
            .ok_or_else(|| format_err!("package {} missing version", name))?
            .to_string();
        let source = pkg.get("source").and_then(|v| v.as_str()).map(String::from);
        let checksum = pkg
            .get("checksum")
            .and_then(|v| v.as_str())
            .map(String::from);
        let dependencies = pkg
            .get("dependencies")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default();

        result.push(LockPackage {
            name,
            version,
            source,
            checksum,
            dependencies,
        });
    }

    Ok(result)
}

/// Get workspace members from root Cargo.toml: returns (crate_name, member_path)
fn get_workspace_members(
    root_toml: &toml::Value,
    workspace_root: &Path,
) -> Result<Vec<(String, PathBuf)>, Error> {
    let members = root_toml
        .get("workspace")
        .and_then(|ws| ws.get("members"))
        .and_then(|m| m.as_array())
        .ok_or_else(|| format_err!("No [workspace] members found in root Cargo.toml"))?;

    let mut result = Vec::new();
    for member in members {
        let member_path = member
            .as_str()
            .ok_or_else(|| format_err!("workspace member is not a string"))?;
        let full_path = workspace_root.join(member_path);
        let cargo_toml = full_path.join("Cargo.toml");
        if cargo_toml.exists() {
            let content = std::fs::read_to_string(&cargo_toml)
                .with_context(|| format!("reading {}", cargo_toml.display()))?;
            let toml: toml::Value = content
                .parse()
                .with_context(|| format!("parsing {}", cargo_toml.display()))?;
            let name = toml
                .get("package")
                .and_then(|p| p.get("name"))
                .and_then(|n| n.as_str())
                .unwrap_or_else(|| {
                    // Fall back to directory name
                    full_path
                        .file_name()
                        .and_then(|n| n.to_str())
                        .unwrap_or("unknown")
                })
                .to_string();
            result.push((name, full_path));
        } else {
            eprintln!(
                "WARNING: workspace member path {} does not contain Cargo.toml",
                full_path.display()
            );
        }
    }

    Ok(result)
}

/// Build an index from Cargo.lock source strings to local paths for git deps
fn build_git_source_index(
    git_sources: &[GitSourceMapping],
    lock_packages: &[LockPackage],
) -> Result<HashMap<String, PathBuf>, Error> {
    let mut index: HashMap<String, PathBuf> = HashMap::new();

    for mapping in git_sources {
        // Find all lock packages that match this git URL + rev
        for pkg in lock_packages {
            if let Some(source) = &pkg.source {
                if source.starts_with("git+") {
                    let source_url = &source[4..]; // strip "git+"
                                                   // Check if this source matches the mapping
                    if source_matches_mapping(source_url, &mapping.url, &mapping.rev) {
                        index.insert(source.clone(), mapping.local_path.clone());
                    }
                }
            }
        }
    }

    Ok(index)
}

/// Check if a Cargo.lock git source URL matches a git source mapping
fn source_matches_mapping(source_url: &str, mapping_url: &str, mapping_rev: &str) -> bool {
    // source_url looks like: "https://github.com/user/repo?tag=v1.0#commitsha"
    // or: "https://github.com/user/repo.git?tag=v1.0#commitsha"
    // mapping_url looks like: "https://github.com/user/repo" or "https://github.com/user/repo.git"
    // mapping_rev looks like: "commitsha"

    // Check if the fragment (commit hash) matches
    if let Some(hash_pos) = source_url.rfind('#') {
        let commit = &source_url[hash_pos + 1..];
        if commit != mapping_rev {
            return false;
        }

        // Check if the base URL matches (strip query params and .git suffix)
        let base = &source_url[..hash_pos];
        let base_no_query = base.split('?').next().unwrap_or(base);

        let normalize = |u: &str| -> String {
            let u = u.trim_end_matches('/');
            let u = u.strip_suffix(".git").unwrap_or(u);
            u.to_lowercase()
        };

        normalize(base_no_query) == normalize(mapping_url)
    } else {
        false
    }
}

/// Make a package ID string matching cargo's format
fn make_package_id(pkg: &LockPackage) -> String {
    match &pkg.source {
        Some(source) => format!("{} {} ({})", pkg.name, pkg.version, source),
        None => format!("path+file:///{}#{}", pkg.name, pkg.version),
    }
}

/// Data extracted from a Cargo.toml manifest
#[derive(Debug, Default)]
struct ManifestData {
    edition: String,
    authors: Vec<String>,
    links: Option<String>,
    features: BTreeMap<String, Vec<String>>,
    normal_deps: Vec<DepSpec>,
    build_deps: Vec<DepSpec>,
    dev_deps: Vec<DepSpec>,
    target_deps: Vec<(String, Vec<DepSpec>, DepKind)>,
    has_lib: Option<bool>,
    lib_name: Option<String>,
    lib_path: Option<String>,
    lib_crate_types: Vec<String>,
    is_proc_macro: bool,
    has_build_rs: bool,
    build_script_path: Option<String>,
    bin_targets: Vec<(String, String)>, // (name, path)
    #[allow(dead_code)]
    default_run: Option<String>,
}

#[derive(Debug, Clone, Copy)]
#[allow(dead_code)]
enum DepKind {
    Normal,
    Build,
    Dev,
}

/// Try to find and read the Cargo.toml for a lock package
fn find_and_read_manifest(
    lock_pkg: &LockPackage,
    member_path: &Option<PathBuf>,
    workspace_root: &Path,
    git_source_index: &HashMap<String, PathBuf>,
    crates_io_manifests: &Option<PathBuf>,
    workspace_package: Option<&toml::Value>,
    workspace_deps: Option<&toml::Value>,
) -> Option<ManifestData> {
    // 1. Workspace member: read from source tree
    if let Some(path) = member_path {
        let cargo_toml = path.join("Cargo.toml");
        if cargo_toml.exists() {
            return read_manifest(
                &cargo_toml,
                Some(workspace_root),
                workspace_package,
                workspace_deps,
            )
            .map_err(|e| {
                eprintln!(
                    "WARNING: Failed to parse workspace member Cargo.toml at {}: {}",
                    cargo_toml.display(),
                    e
                );
                e
            })
            .ok();
        }
    }

    // 2. Git dependency: read from pre-fetched source
    if let Some(source) = &lock_pkg.source {
        if source.starts_with("git+") {
            if let Some(git_root) = git_source_index.get(source) {
                // Find the crate within the git repo
                // It could be the root or a workspace member
                let manifest = find_crate_in_git_source(git_root, &lock_pkg.name);
                if let Some(manifest_path) = manifest {
                    // Read workspace root Cargo.toml for workspace inheritance
                    let (ws_pkg, ws_deps) = read_workspace_context(git_root);
                    return read_manifest(
                        &manifest_path,
                        Some(git_root),
                        ws_pkg.as_ref(),
                        ws_deps.as_ref(),
                    )
                    .map_err(|e| {
                        eprintln!(
                            "WARNING: Failed to parse git dep Cargo.toml at {}: {}",
                            manifest_path.display(),
                            e
                        );
                        e
                    })
                    .ok();
                }
            }
        }
    }

    // 3. Crates.io: read from extracted manifests directory
    if let Some(source) = &lock_pkg.source {
        if source.starts_with("registry+") || source.starts_with("sparse+") {
            if let Some(manifests_dir) = crates_io_manifests {
                let cargo_toml = manifests_dir
                    .join(&lock_pkg.name)
                    .join(&lock_pkg.version)
                    .join("Cargo.toml");
                if cargo_toml.exists() {
                    return read_manifest(&cargo_toml, None, None, None)
                        .map_err(|e| {
                            eprintln!(
                                "WARNING: Failed to parse crates.io Cargo.toml at {}: {}",
                                cargo_toml.display(),
                                e
                            );
                            e
                        })
                        .ok();
                }
            }
        }
    }

    None
}

/// Find a crate's Cargo.toml within a git source directory
fn find_crate_in_git_source(git_root: &Path, crate_name: &str) -> Option<PathBuf> {
    // Check root Cargo.toml first
    let root_toml = git_root.join("Cargo.toml");
    if root_toml.exists() {
        if let Ok(content) = std::fs::read_to_string(&root_toml) {
            if let Ok(toml) = content.parse::<toml::Value>() {
                // Check if this is the crate itself
                if let Some(name) = toml
                    .get("package")
                    .and_then(|p| p.get("name"))
                    .and_then(|n| n.as_str())
                {
                    if name == crate_name {
                        return Some(root_toml);
                    }
                }

                // Check workspace members
                if let Some(members) = toml
                    .get("workspace")
                    .and_then(|ws| ws.get("members"))
                    .and_then(|m| m.as_array())
                {
                    for member in members {
                        if let Some(member_str) = member.as_str() {
                            // Handle glob patterns (e.g., "crates/*")
                            if member_str.contains('*') {
                                let prefix = member_str.split('*').next().unwrap_or("");
                                let prefix_path = git_root.join(prefix);
                                if prefix_path.exists() {
                                    if let Ok(entries) = std::fs::read_dir(&prefix_path) {
                                        for entry in entries.flatten() {
                                            let candidate = entry.path().join("Cargo.toml");
                                            if candidate.exists() {
                                                if let Some(found) =
                                                    check_manifest_name(&candidate, crate_name)
                                                {
                                                    return Some(found);
                                                }
                                            }
                                        }
                                    }
                                }
                            } else {
                                let candidate = git_root.join(member_str).join("Cargo.toml");
                                if candidate.exists() {
                                    if let Some(found) = check_manifest_name(&candidate, crate_name)
                                    {
                                        return Some(found);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // Fallback: search recursively (limited depth)
    find_manifest_recursive(git_root, crate_name, 4)
}

/// Read workspace [workspace.package] and [workspace.dependencies] from the root Cargo.toml
fn read_workspace_context(workspace_root: &Path) -> (Option<toml::Value>, Option<toml::Value>) {
    let root_toml_path = workspace_root.join("Cargo.toml");
    if !root_toml_path.exists() {
        return (None, None);
    }
    let content = match std::fs::read_to_string(&root_toml_path) {
        Ok(c) => c,
        Err(_) => return (None, None),
    };
    let toml: toml::Value = match content.parse() {
        Ok(t) => t,
        Err(_) => return (None, None),
    };
    let ws = toml.get("workspace");
    let ws_pkg = ws.and_then(|w| w.get("package")).cloned();
    let ws_deps = ws.and_then(|w| w.get("dependencies")).cloned();
    (ws_pkg, ws_deps)
}

fn check_manifest_name(cargo_toml: &Path, expected_name: &str) -> Option<PathBuf> {
    let content = std::fs::read_to_string(cargo_toml).ok()?;
    let toml: toml::Value = content.parse().ok()?;
    let name = toml.get("package")?.get("name")?.as_str()?;
    if name == expected_name {
        Some(cargo_toml.to_path_buf())
    } else {
        None
    }
}

fn find_manifest_recursive(dir: &Path, crate_name: &str, max_depth: u32) -> Option<PathBuf> {
    if max_depth == 0 {
        return None;
    }
    let cargo_toml = dir.join("Cargo.toml");
    if cargo_toml.exists() {
        if let Some(found) = check_manifest_name(&cargo_toml, crate_name) {
            return Some(found);
        }
    }
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                // Skip common non-source directories
                let dirname = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
                if dirname.starts_with('.') || dirname == "target" || dirname == "node_modules" {
                    continue;
                }
                if let Some(found) = find_manifest_recursive(&path, crate_name, max_depth - 1) {
                    return Some(found);
                }
            }
        }
    }
    None
}

/// Read and parse a Cargo.toml into ManifestData
fn read_manifest(
    cargo_toml_path: &Path,
    _workspace_root: Option<&Path>,
    workspace_package: Option<&toml::Value>,
    workspace_deps: Option<&toml::Value>,
) -> Result<ManifestData, Error> {
    let content = std::fs::read_to_string(cargo_toml_path)?;
    let toml: toml::Value = content.parse()?;

    let package = toml.get("package");

    let resolve_ws_string = |key: &str| -> String {
        if let Some(pkg) = package {
            if let Some(val) = pkg.get(key) {
                if let Some(s) = val.as_str() {
                    return s.to_string();
                }
                // Check for workspace inheritance: { workspace = true }
                if val.get("workspace").and_then(|w| w.as_bool()) == Some(true) {
                    if let Some(ws) = workspace_package {
                        if let Some(s) = ws.get(key).and_then(|v| v.as_str()) {
                            return s.to_string();
                        }
                    }
                }
            }
        }
        String::new()
    };

    let edition = {
        let e = resolve_ws_string("edition");
        if e.is_empty() {
            "2015".to_string()
        } else {
            e
        }
    };

    let authors = {
        let mut result = Vec::new();
        if let Some(pkg) = package {
            if let Some(arr) = pkg.get("authors").and_then(|v| v.as_array()) {
                for a in arr {
                    if let Some(s) = a.as_str() {
                        result.push(s.to_string());
                    }
                }
            } else if pkg
                .get("authors")
                .and_then(|v| v.get("workspace"))
                .and_then(|w| w.as_bool())
                == Some(true)
            {
                if let Some(ws) = workspace_package {
                    if let Some(arr) = ws.get("authors").and_then(|v| v.as_array()) {
                        for a in arr {
                            if let Some(s) = a.as_str() {
                                result.push(s.to_string());
                            }
                        }
                    }
                }
            }
        }
        result
    };

    let links = package
        .and_then(|p| p.get("links"))
        .and_then(|v| v.as_str())
        .map(String::from);

    // Parse features
    let features = parse_features(&toml);

    // Parse dependencies
    let normal_deps = parse_dep_table(toml.get("dependencies"), workspace_deps);
    let build_deps = parse_dep_table(toml.get("build-dependencies"), workspace_deps);
    let dev_deps = parse_dep_table(toml.get("dev-dependencies"), workspace_deps);

    // Parse target-specific dependencies
    let mut target_deps = Vec::new();
    if let Some(targets) = toml.get("target").and_then(|t| t.as_table()) {
        for (target_cfg, target_val) in targets {
            let t_normal = parse_dep_table(target_val.get("dependencies"), workspace_deps);
            if !t_normal.is_empty() {
                target_deps.push((target_cfg.clone(), t_normal, DepKind::Normal));
            }
            let t_build = parse_dep_table(target_val.get("build-dependencies"), workspace_deps);
            if !t_build.is_empty() {
                target_deps.push((target_cfg.clone(), t_build, DepKind::Build));
            }
            let t_dev = parse_dep_table(target_val.get("dev-dependencies"), workspace_deps);
            if !t_dev.is_empty() {
                target_deps.push((target_cfg.clone(), t_dev, DepKind::Dev));
            }
        }
    }

    // Parse targets (lib, bin, build.rs)
    let crate_dir = cargo_toml_path.parent().unwrap_or(Path::new("."));

    let (has_lib, lib_name, lib_path, lib_crate_types, is_proc_macro) =
        parse_lib_target(&toml, crate_dir);
    let (has_build_rs, build_script_path) = {
        let build_val = toml.get("package").and_then(|p| p.get("build"));
        match build_val {
            Some(b) => {
                if let Some(s) = b.as_str() {
                    (
                        !s.is_empty(),
                        if s.is_empty() {
                            None
                        } else {
                            Some(s.to_string())
                        },
                    )
                } else {
                    (b.as_bool().unwrap_or(false), None)
                }
            }
            None => (crate_dir.join("build.rs").exists(), None),
        }
    };

    let bin_targets = parse_bin_targets(&toml, crate_dir);
    let default_run = package
        .and_then(|p| p.get("default-run"))
        .and_then(|v| v.as_str())
        .map(String::from);

    Ok(ManifestData {
        edition,
        authors,
        links,
        features,
        normal_deps,
        build_deps,
        dev_deps,
        target_deps,
        has_lib,
        lib_name,
        lib_path,
        lib_crate_types,
        is_proc_macro,
        has_build_rs,
        build_script_path,
        bin_targets,
        default_run,
    })
}

/// Parse [features] table from Cargo.toml
fn parse_features(toml: &toml::Value) -> BTreeMap<String, Vec<String>> {
    let mut result = BTreeMap::new();
    if let Some(features) = toml.get("features").and_then(|f| f.as_table()) {
        for (name, values) in features {
            let mut feature_list = Vec::new();
            if let Some(arr) = values.as_array() {
                for v in arr {
                    if let Some(s) = v.as_str() {
                        feature_list.push(s.to_string());
                    }
                }
            }
            result.insert(name.clone(), feature_list);
        }
    }
    result
}

/// Parse a dependency table from Cargo.toml
fn parse_dep_table(
    table: Option<&toml::Value>,
    workspace_deps: Option<&toml::Value>,
) -> Vec<DepSpec> {
    let mut result = Vec::new();
    let table = match table.and_then(|t| t.as_table()) {
        Some(t) => t,
        None => return result,
    };

    for (name, value) in table {
        let dep = parse_single_dep(name, value, workspace_deps);
        result.push(dep);
    }

    result
}

/// Parse a single dependency entry from Cargo.toml
fn parse_single_dep(
    name: &str,
    value: &toml::Value,
    workspace_deps: Option<&toml::Value>,
) -> DepSpec {
    // Simple version string: dep = "1.0"
    if let Some(version) = value.as_str() {
        return DepSpec {
            name: name.to_string(),
            package: None,
            version_req: version.to_string(),
            optional: false,
            default_features: true,
            features: Vec::new(),
            source: None,
        };
    }

    // Table form: dep = { version = "1.0", features = [...], ... }
    if let Some(table) = value.as_table() {
        // Handle workspace inheritance
        if table.get("workspace").and_then(|w| w.as_bool()) == Some(true) {
            if let Some(ws_deps) = workspace_deps {
                if let Some(ws_dep) = ws_deps.get(name) {
                    let mut base = parse_single_dep(name, ws_dep, None);
                    // Override with local settings
                    if let Some(features) = table.get("features").and_then(|f| f.as_array()) {
                        let extra: Vec<String> = features
                            .iter()
                            .filter_map(|f| f.as_str().map(String::from))
                            .collect();
                        base.features.extend(extra);
                    }
                    if let Some(optional) = table.get("optional").and_then(|o| o.as_bool()) {
                        base.optional = optional;
                    }
                    if let Some(df) = table
                        .get("default-features")
                        .or_else(|| table.get("default_features"))
                        .and_then(|d| d.as_bool())
                    {
                        base.default_features = df;
                    }
                    return base;
                }
            }
            // Workspace dep not found, use defaults
            return DepSpec {
                name: name.to_string(),
                package: None,
                version_req: "*".to_string(),
                optional: false,
                default_features: true,
                features: Vec::new(),
                source: None,
            };
        }

        let package = table
            .get("package")
            .and_then(|p| p.as_str())
            .map(String::from);
        let version_req = table
            .get("version")
            .and_then(|v| v.as_str())
            .unwrap_or("*")
            .to_string();
        let optional = table
            .get("optional")
            .and_then(|o| o.as_bool())
            .unwrap_or(false);
        let default_features = table
            .get("default-features")
            .and_then(|d| d.as_bool())
            .unwrap_or(true);
        let features = table
            .get("features")
            .and_then(|f| f.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default();

        // Build source hint from git/path info, including tag/branch/rev for disambiguation
        let source = if let Some(git) = table.get("git").and_then(|g| g.as_str()) {
            let mut s = format!("git+{}", git);
            if let Some(tag) = table.get("tag").and_then(|t| t.as_str()) {
                s.push_str(&format!("?tag={}", tag));
            } else if let Some(branch) = table.get("branch").and_then(|b| b.as_str()) {
                s.push_str(&format!("?branch={}", branch));
            } else if let Some(rev) = table.get("rev").and_then(|r| r.as_str()) {
                s.push_str(&format!("?rev={}", rev));
            }
            Some(s)
        } else {
            None
        };

        return DepSpec {
            name: name.to_string(),
            package,
            version_req,
            optional,
            default_features,
            features,
            source,
        };
    }

    // Fallback
    DepSpec {
        name: name.to_string(),
        package: None,
        version_req: "*".to_string(),
        optional: false,
        default_features: true,
        features: Vec::new(),
        source: None,
    }
}

/// Parse [lib] target from Cargo.toml
/// Returns (has_lib, lib_name, lib_path, lib_crate_types, is_proc_macro)
fn parse_lib_target(
    toml: &toml::Value,
    crate_dir: &Path,
) -> (
    Option<bool>,
    Option<String>,
    Option<String>,
    Vec<String>,
    bool,
) {
    if let Some(lib) = toml.get("lib") {
        let name = lib.get("name").and_then(|n| n.as_str()).map(String::from);
        let path = lib.get("path").and_then(|p| p.as_str()).map(String::from);
        let crate_types: Vec<String> = lib
            .get("crate-type")
            .or_else(|| lib.get("crate_type"))
            .and_then(|c| c.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default();
        let proc_macro = lib
            .get("proc-macro")
            .or_else(|| lib.get("proc_macro"))
            .and_then(|p| p.as_bool())
            .unwrap_or(false)
            || crate_types.iter().any(|t| t == "proc-macro");
        return (Some(true), name, path, crate_types, proc_macro);
    }

    // Default: check if src/lib.rs exists.
    // If it doesn't exist on disk (e.g. crates.io manifest-only directory),
    // still assume has_lib = true since the build system will check at build time.
    // This is safe: if src/lib.rs truly doesn't exist, the builder skips the lib.
    let has_lib = if crate_dir.join("src").join("lib.rs").exists() {
        Some(true)
    } else if !crate_dir.join("src").exists() {
        // The source directory doesn't exist here (manifest-only), assume lib exists
        Some(true)
    } else {
        None
    };
    (has_lib, None, None, Vec::new(), false)
}

/// Parse [[bin]] targets from Cargo.toml
fn parse_bin_targets(toml: &toml::Value, crate_dir: &Path) -> Vec<(String, String)> {
    let mut bins = Vec::new();

    if let Some(bin_arr) = toml.get("bin").and_then(|b| b.as_array()) {
        for bin in bin_arr {
            let name = bin
                .get("name")
                .and_then(|n| n.as_str())
                .unwrap_or("unknown");
            let path = bin
                .get("path")
                .and_then(|p| p.as_str())
                .unwrap_or("src/main.rs");
            bins.push((name.to_string(), path.to_string()));
        }
    } else {
        // Default binary target detection
        let main_rs = crate_dir.join("src").join("main.rs");
        if main_rs.exists() {
            let name = toml
                .get("package")
                .and_then(|p| p.get("name"))
                .and_then(|n| n.as_str())
                .unwrap_or("unknown");
            bins.push((name.to_string(), "src/main.rs".to_string()));
        }
    }

    bins
}

/// Build a Package JSON value for cargo_metadata
fn build_package_json(
    lock_pkg: &LockPackage,
    pkg_id: &str,
    workspace_root: &Path,
    member_path: &Option<PathBuf>,
    manifest_data: &Option<ManifestData>,
) -> Result<(serde_json::Value, Vec<(DepSpec, DepKind, Option<String>)>), Error> {
    let manifest_path = if let Some(path) = member_path {
        path.join("Cargo.toml")
    } else {
        // For non-workspace members, use a synthetic path
        workspace_root
            .join("target")
            .join("lockfile-metadata")
            .join(&lock_pkg.name)
            .join(&lock_pkg.version)
            .join("Cargo.toml")
    };

    let edition = manifest_data
        .as_ref()
        .map(|m| m.edition.as_str())
        .unwrap_or("2015");

    let authors: Vec<String> = manifest_data
        .as_ref()
        .map(|m| m.authors.clone())
        .unwrap_or_default();

    let links = manifest_data.as_ref().and_then(|m| m.links.clone());

    let features: BTreeMap<String, Vec<String>> = manifest_data
        .as_ref()
        .map(|m| m.features.clone())
        .unwrap_or_default();

    let source = &lock_pkg.source;

    // Build dependencies list for Package
    let mut all_dep_specs: Vec<(DepSpec, DepKind, Option<String>)> = Vec::new();
    let mut deps_json = Vec::new();

    if let Some(manifest) = manifest_data {
        // We have manifest data - use it
        for dep in &manifest.normal_deps {
            let dep_json = dep_spec_to_json(dep, "normal", None);
            deps_json.push(dep_json);
            all_dep_specs.push((dep.clone(), DepKind::Normal, None));
        }
        for dep in &manifest.build_deps {
            let dep_json = dep_spec_to_json(dep, "build", None);
            deps_json.push(dep_json);
            all_dep_specs.push((dep.clone(), DepKind::Build, None));
        }
        for dep in &manifest.dev_deps {
            let dep_json = dep_spec_to_json(dep, "dev", None);
            deps_json.push(dep_json);
            all_dep_specs.push((dep.clone(), DepKind::Dev, None));
        }
        for (target, target_deps, kind) in &manifest.target_deps {
            let kind_str = match kind {
                DepKind::Normal => "normal",
                DepKind::Build => "build",
                DepKind::Dev => "dev",
            };
            for dep in target_deps {
                let dep_json = dep_spec_to_json(dep, kind_str, Some(target));
                deps_json.push(dep_json);
                all_dep_specs.push((dep.clone(), *kind, Some(target.clone())));
            }
        }
    } else {
        // No manifest data - derive minimal deps from Cargo.lock
        // All deps are treated as normal kind
        for dep_str in &lock_pkg.dependencies {
            let (dep_name, dep_version, dep_source) = parse_lock_dep_string(dep_str);
            let version_req = if !dep_version.is_empty() {
                format!("={}", dep_version)
            } else {
                "*".to_string()
            };
            let dep = DepSpec {
                name: dep_name,
                package: None,
                version_req,
                optional: false,
                default_features: true,
                features: Vec::new(),
                source: dep_source,
            };
            let dep_json = dep_spec_to_json(&dep, "normal", None);
            deps_json.push(dep_json);
            all_dep_specs.push((dep, DepKind::Normal, None));
        }
    }

    // Build targets
    let mut targets_json = Vec::new();

    if let Some(manifest) = manifest_data {
        let has_lib = manifest.has_lib.unwrap_or(false);
        if has_lib {
            let lib_name = manifest
                .lib_name
                .as_deref()
                .unwrap_or(&lock_pkg.name)
                .replace('-', "_");

            let mut kind = vec!["lib"];
            let mut crate_types = vec!["lib"];
            if manifest.is_proc_macro {
                kind = vec!["proc-macro"];
                crate_types = vec!["proc-macro"];
            } else if !manifest.lib_crate_types.is_empty() {
                kind = manifest
                    .lib_crate_types
                    .iter()
                    .map(|s| s.as_str())
                    .collect();
                crate_types = kind.clone();
            }

            let lib_src_path = manifest.lib_path.as_deref().unwrap_or("src/lib.rs");
            targets_json.push(json!({
                "kind": kind,
                "crate_types": crate_types,
                "name": lib_name,
                "src_path": manifest_path.parent().unwrap_or(Path::new(".")).join(lib_src_path).to_string_lossy(),
                "edition": edition,
                "doctest": true,
            }));
        }

        if manifest.has_build_rs {
            let build_src_path = manifest.build_script_path.as_deref().unwrap_or("build.rs");
            targets_json.push(json!({
                "kind": ["custom-build"],
                "crate_types": ["bin"],
                "name": "build-script-build",
                "src_path": manifest_path.parent().unwrap_or(Path::new(".")).join(build_src_path).to_string_lossy(),
                "edition": edition,
                "doctest": false,
            }));
        }

        for (bin_name, bin_path) in &manifest.bin_targets {
            targets_json.push(json!({
                "kind": ["bin"],
                "crate_types": ["bin"],
                "name": bin_name,
                "src_path": manifest_path.parent().unwrap_or(Path::new(".")).join(bin_path).to_string_lossy(),
                "edition": edition,
                "doctest": false,
            }));
        }
    } else {
        // Default: assume lib target
        targets_json.push(json!({
            "kind": ["lib"],
            "crate_types": ["lib"],
            "name": lock_pkg.name.replace('-', "_"),
            "src_path": manifest_path.parent().unwrap_or(Path::new(".")).join("src/lib.rs").to_string_lossy(),
            "edition": edition,
            "doctest": true,
        }));
    }

    // If no targets, add a default lib
    if targets_json.is_empty() {
        targets_json.push(json!({
            "kind": ["lib"],
            "crate_types": ["lib"],
            "name": lock_pkg.name.replace('-', "_"),
            "src_path": manifest_path.parent().unwrap_or(Path::new(".")).join("src/lib.rs").to_string_lossy(),
            "edition": edition,
            "doctest": true,
        }));
    }

    let pkg_json = json!({
        "name": lock_pkg.name,
        "version": lock_pkg.version,
        "id": pkg_id,
        "source": source,
        "dependencies": deps_json,
        "targets": targets_json,
        "features": features,
        "manifest_path": manifest_path.to_string_lossy(),
        "edition": edition,
        "authors": authors,
        "links": links,
    });

    Ok((pkg_json, all_dep_specs))
}

/// Convert a DepSpec to JSON matching cargo_metadata::Dependency format
fn dep_spec_to_json(dep: &DepSpec, kind: &str, target: Option<&str>) -> serde_json::Value {
    let kind_json = match kind {
        "normal" => serde_json::Value::Null,
        other => json!(other),
    };

    json!({
        "name": dep.package.as_deref().unwrap_or(&dep.name),
        "source": dep.source,
        "req": dep.version_req,
        "kind": kind_json,
        "rename": if dep.package.is_some() { Some(&dep.name) } else { None },
        "optional": dep.optional,
        "uses_default_features": dep.default_features,
        "features": dep.features,
        "target": target,
        "registry": null,
    })
}

/// Build a Node JSON value for the resolve graph
fn build_node_json(
    lock_pkg: &LockPackage,
    pkg_id: &str,
    dep_specs: &[(DepSpec, DepKind, Option<String>)],
    all_lock_packages: &[LockPackage],
    all_pkg_ids: &[String],
    lock_pkg_lookup: &HashMap<(String, String, Option<String>), usize>,
) -> Result<serde_json::Value, Error> {
    let mut dep_ids = Vec::new();
    let mut deps_json = Vec::new();

    // Match each dependency in Cargo.lock to its resolved package.
    // Cargo.lock dep strings can be:
    //   "name" (unique name in lockfile)
    //   "name version" (ambiguous name, version disambiguates)
    //   "name version (source)" (same version from different sources)
    for dep_str in &lock_pkg.dependencies {
        let (dep_name, dep_version, dep_source) = parse_lock_dep_string(dep_str);

        let resolved_idx: Option<usize> = if !dep_version.is_empty() {
            // Have name + version (and possibly source)
            lock_pkg_lookup
                .get(&(dep_name.clone(), dep_version.clone(), dep_source.clone()))
                .or_else(|| lock_pkg_lookup.get(&(dep_name.clone(), dep_version.clone(), None)))
                .copied()
                .or_else(|| {
                    all_lock_packages
                        .iter()
                        .enumerate()
                        .find(|(_, p)| p.name == dep_name && p.version == dep_version)
                        .map(|(i, _)| i)
                })
        } else {
            // Just a name - find the unique package with this name
            let matches: Vec<usize> = all_lock_packages
                .iter()
                .enumerate()
                .filter(|(_, p)| p.name == dep_name)
                .map(|(i, _)| i)
                .collect();
            if matches.len() == 1 {
                Some(matches[0])
            } else {
                // Ambiguous - should not happen with just a name in Cargo.lock
                // but try to pick one
                matches.into_iter().next()
            }
        };

        if let Some(idx) = resolved_idx {
            let resolved_id = &all_pkg_ids[idx];
            dep_ids.push(resolved_id.clone());

            let actual_name = all_lock_packages[idx].name.clone();
            let resolved_version = all_lock_packages[idx].version.clone();
            let resolved_source = all_lock_packages[idx].source.clone();
            let (dep_kind_info, node_dep_name) = find_dep_kind_and_name(
                &actual_name,
                &resolved_version,
                &resolved_source,
                dep_specs,
            );

            deps_json.push(json!({
                "name": node_dep_name.unwrap_or_else(|| actual_name.clone()).replace('-', "_"),
                "pkg": resolved_id,
                "dep_kinds": dep_kind_info,
                "_pkg_name": actual_name,
                "_pkg_version": resolved_version,
                "_pkg_source": resolved_source,
            }));
        } else {
            eprintln!(
                "WARNING: Could not resolve dependency '{}' of {} {}",
                dep_str, lock_pkg.name, lock_pkg.version
            );
        }
    }

    // Post-processing: fix name assignments for packages with multiple versions.
    // Group deps by package name and reassign using scoring-based matching.
    let mut pkg_groups: HashMap<String, Vec<usize>> = HashMap::new();
    for (i, dep) in deps_json.iter().enumerate() {
        let pkg_name = dep["_pkg_name"].as_str().unwrap_or("").to_string();
        pkg_groups.entry(pkg_name).or_default().push(i);
    }

    for (pkg_name, dep_indices) in &pkg_groups {
        if dep_indices.len() <= 1 {
            continue; // No ambiguity for single deps
        }
        // Find all dep specs for this package name
        let specs_for_pkg: Vec<&(DepSpec, DepKind, Option<String>)> = dep_specs
            .iter()
            .filter(|(spec, _, _)| {
                let sn = spec.package.as_deref().unwrap_or(&spec.name);
                sn == pkg_name
            })
            .collect();

        if specs_for_pkg.len() < dep_indices.len() {
            continue; // Not enough specs to assign
        }

        // Score each (dep_index, spec) pair
        let mut scores: Vec<(usize, usize, i32)> = Vec::new(); // (dep_idx, spec_idx, score)
        for &di in dep_indices {
            let dep_source = deps_json[di]["_pkg_source"].as_str().unwrap_or("");
            let dep_version = deps_json[di]["_pkg_version"].as_str().unwrap_or("");
            for (si, (spec, _, _)) in specs_for_pkg.iter().enumerate() {
                let mut score = 0i32;

                // Version compatibility bonus (critical for disambiguating
                // same-package deps like yamux012 vs yamux013)
                let version_req = &spec.version_req;
                if version_req != "*" && !dep_version.is_empty() {
                    let version_matches = if let Some(exact) = version_req.strip_prefix('=') {
                        dep_version == exact
                    } else if let (Ok(req), Ok(ver)) = (
                        semver::VersionReq::parse(version_req),
                        semver::Version::parse(dep_version),
                    ) {
                        req.matches(&ver)
                    } else {
                        false
                    };
                    if version_matches {
                        score += 200; // Version match is the strongest signal
                    }
                }

                // Source tag match: both have similar tag keywords
                if let Some(spec_source) = &spec.source {
                    if !dep_source.is_empty() && dep_source.starts_with(spec_source) {
                        score += 100; // Exact source prefix match
                    } else if !dep_source.is_empty() {
                        // Check if both sources share the same tag keyword pattern
                        let dep_has_hf = dep_source.contains("hard-fork-test")
                            || dep_source.contains("hard_fork");
                        let spec_has_hf = spec_source.contains("hard-fork-test")
                            || spec_source.contains("hard_fork");
                        if dep_has_hf && spec_has_hf {
                            score += 50; // Both are hard-fork variants
                        } else if !dep_has_hf && !spec_has_hf {
                            score += 30; // Both are non-hard-fork git variants
                        }
                    }
                } else {
                    // Spec has no source (version-only dep).
                    // Prefer matching to registry packages or non-HF git packages
                    if dep_source.starts_with("registry+") || dep_source.starts_with("sparse+") {
                        score += 80; // Registry package matches version-only dep
                    } else if !dep_source.is_empty()
                        && !dep_source.contains("hard-fork-test")
                        && !dep_source.contains("hard_fork")
                    {
                        score += 20; // Non-HF git package, version-only dep (likely [patch] redirect)
                    }
                }
                scores.push((di, si, score));
            }
        }

        // Greedy assignment: pick highest-scoring pairs, avoiding reuse
        scores.sort_by(|a, b| b.2.cmp(&a.2));
        let mut used_deps: std::collections::HashSet<usize> = std::collections::HashSet::new();
        let mut used_specs: std::collections::HashSet<usize> = std::collections::HashSet::new();

        for (di, si, _score) in &scores {
            if used_deps.contains(di) || used_specs.contains(si) {
                continue;
            }
            let spec = &specs_for_pkg[*si].0;
            let new_name = if spec.package.is_some() {
                spec.name.replace('-', "_")
            } else {
                pkg_name.replace('-', "_")
            };
            deps_json[*di]["name"] = json!(new_name);
            used_deps.insert(*di);
            used_specs.insert(*si);
        }
    }

    // Clean up internal fields
    for dep in &mut deps_json {
        if let Some(obj) = dep.as_object_mut() {
            obj.remove("_pkg_name");
            obj.remove("_pkg_version");
            obj.remove("_pkg_source");
        }
    }

    // Resolve features: for --all-features mode, enable all features
    let all_features: Vec<String> = Vec::new(); // Will be populated during resolution

    Ok(json!({
        "id": pkg_id,
        "dependencies": dep_ids,
        "deps": deps_json,
        "features": all_features,
    }))
}

/// Find the dependency kind info and the node dep name for a dependency.
///
/// Returns (dep_kinds_json, node_dep_name). The node_dep_name is the rename
/// target for renamed deps, which is critical for disambiguation in resolve.rs.
/// When the resolved package's source matches a dep spec's source (e.g., both
/// point to the same git URL+tag), we use that spec's key name.
fn find_dep_kind_and_name(
    dep_name: &str,
    dep_version: &str,
    dep_source: &Option<String>,
    dep_specs: &[(DepSpec, DepKind, Option<String>)],
) -> (Vec<serde_json::Value>, Option<String>) {
    let matching: Vec<_> = dep_specs
        .iter()
        .filter(|(spec, _, _)| {
            let spec_name = spec.package.as_deref().unwrap_or(&spec.name);
            spec_name == dep_name || spec.name == dep_name
        })
        .collect();

    if matching.is_empty() {
        return (vec![json!({"kind": null, "target": null})], None);
    }

    // If there's exactly one match, use it directly.
    // If there are multiple matches (same package from different sources),
    // try to pick the best one using source, then version matching.
    let best_match = if matching.len() == 1 {
        Some(matching[0])
    } else {
        // Strategy 1: Try to match by source string (for git deps)
        let by_source = dep_source.as_ref().and_then(|pkg_source| {
            matching
                .iter()
                .find(|(spec, _, _)| {
                    if let Some(spec_source) = &spec.source {
                        pkg_source.starts_with(spec_source)
                    } else {
                        false
                    }
                })
                .copied()
        });

        if by_source.is_some() {
            by_source
        } else {
            // Strategy 2: Try version matching.
            // Compare the dep spec's version_req against the resolved version.
            // For exact versions like "=8.0.2", match against "8.0.2".
            // For semver ranges like "^2.0.0", check if version is compatible.
            let by_version = matching
                .iter()
                .find(|(spec, _, _)| {
                    let req = &spec.version_req;
                    if req == "*" {
                        false // wildcard doesn't help disambiguate
                    } else if let Some(exact) = req.strip_prefix('=') {
                        dep_version == exact
                    } else {
                        // Try semver matching
                        if let (Ok(req), Ok(ver)) = (
                            semver::VersionReq::parse(req),
                            semver::Version::parse(dep_version),
                        ) {
                            req.matches(&ver)
                        } else {
                            false
                        }
                    }
                })
                .copied();

            if by_version.is_some() {
                by_version
            } else if let Some(pkg_source) = dep_source {
                if pkg_source.starts_with("registry+") || pkg_source.starts_with("sparse+") {
                    // Strategy 3: For registry packages, prefer spec WITHOUT source
                    matching
                        .iter()
                        .find(|(spec, _, _)| spec.source.is_none())
                        .copied()
                } else {
                    // Strategy 4: For git packages, check base URL match.
                    // Only use base URL match if exactly ONE spec matches,
                    // otherwise it's ambiguous (same repo, different tags).
                    let pkg_base_url = pkg_source.split('?').next().unwrap_or(pkg_source);
                    let base_matches: Vec<_> = matching
                        .iter()
                        .filter(|(spec, _, _)| {
                            if let Some(spec_source) = &spec.source {
                                let spec_base =
                                    spec_source.split('?').next().unwrap_or(spec_source);
                                spec_base == pkg_base_url
                            } else {
                                false
                            }
                        })
                        .collect();

                    if base_matches.len() == 1 {
                        Some(*base_matches[0])
                    } else {
                        // Multiple specs share the base URL or none matched.
                        // Pick the spec without a source hint (version-only dep
                        // that was likely redirected via [patch.crates-io]).
                        matching
                            .iter()
                            .find(|(spec, _, _)| spec.source.is_none())
                            .copied()
                    }
                }
            } else {
                matching
                    .iter()
                    .find(|(spec, _, _)| spec.source.is_none())
                    .copied()
            }
        }
    };

    // Determine the node dep name from the best matching spec
    let node_dep_name = best_match.and_then(|(spec, _, _)| {
        // If spec has a `package` field, then `spec.name` is the rename target
        if spec.package.is_some() {
            Some(spec.name.clone())
        } else {
            None
        }
    });

    let kinds = matching
        .iter()
        .map(|(_, kind, target)| {
            let kind_json = match kind {
                DepKind::Normal => serde_json::Value::Null,
                DepKind::Build => json!("build"),
                DepKind::Dev => json!("dev"),
            };
            json!({
                "kind": kind_json,
                "target": target,
            })
        })
        .collect();

    (kinds, node_dep_name)
}

/// Parse a Cargo.lock dependency string like "name version (source)"
fn parse_lock_dep_string(s: &str) -> (String, String, Option<String>) {
    let mut parts = s.splitn(3, ' ');
    let name = parts.next().unwrap_or("").to_string();
    let version = parts.next().unwrap_or("").to_string();
    let source = parts.next().map(|s| {
        // Remove parentheses: "(source)" -> "source"
        s.trim_start_matches('(').trim_end_matches(')').to_string()
    });
    (name, version, source)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_lock_dep_string() {
        let (name, version, source) = parse_lock_dep_string("serde 1.0.123");
        assert_eq!(name, "serde");
        assert_eq!(version, "1.0.123");
        assert_eq!(source, None);

        let (name, version, source) = parse_lock_dep_string(
            "serde 1.0.123 (registry+https://github.com/rust-lang/crates.io-index)",
        );
        assert_eq!(name, "serde");
        assert_eq!(version, "1.0.123");
        assert_eq!(
            source,
            Some("registry+https://github.com/rust-lang/crates.io-index".to_string())
        );
    }

    #[test]
    fn test_source_matches_mapping() {
        assert!(source_matches_mapping(
            "https://github.com/user/repo?tag=v1.0#abc123",
            "https://github.com/user/repo",
            "abc123"
        ));

        assert!(source_matches_mapping(
            "https://github.com/user/repo.git?tag=v1.0#abc123",
            "https://github.com/user/repo",
            "abc123"
        ));

        assert!(!source_matches_mapping(
            "https://github.com/user/repo?tag=v1.0#abc123",
            "https://github.com/user/repo",
            "def456"
        ));
    }

    #[test]
    fn test_parse_features() {
        let toml: toml::Value = r#"
            [features]
            default = ["std"]
            std = ["serde/std"]
            alloc = []
        "#
        .parse()
        .unwrap();

        let features = parse_features(&toml);
        assert_eq!(features.len(), 3);
        assert_eq!(features["default"], vec!["std"]);
        assert_eq!(features["std"], vec!["serde/std"]);
        assert!(features["alloc"].is_empty());
    }

    #[test]
    fn test_make_package_id() {
        let pkg = LockPackage {
            name: "serde".to_string(),
            version: "1.0.123".to_string(),
            source: Some("registry+https://github.com/rust-lang/crates.io-index".to_string()),
            checksum: None,
            dependencies: Vec::new(),
        };
        assert_eq!(
            make_package_id(&pkg),
            "serde 1.0.123 (registry+https://github.com/rust-lang/crates.io-index)"
        );

        let local_pkg = LockPackage {
            name: "my-crate".to_string(),
            version: "0.1.0".to_string(),
            source: None,
            checksum: None,
            dependencies: Vec::new(),
        };
        assert_eq!(make_package_id(&local_pkg), "path+file:///my-crate#0.1.0");
    }
}
