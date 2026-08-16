use super::forest_meta::{region_size_po2, RegionForestMeta, META_FILE_NAME};
use super::{RegionError, RegionFile, RegionFormat};
use crate::constants::voxel_constants::MAX_LOD;
use crate::math::Vector3i;
use crate::storage::voxel_buffer::{ALL_CHANNELS_MASK, MAX_CHANNELS};
use crate::storage::VoxelBuffer;
use crate::streams::compressed_data::Compression;
use crate::streams::{
    LoadResult, SaveMode, StreamResult, VoxelLoadQuery, VoxelSaveQuery, VoxelStream,
    VoxelStreamError,
};
use std::collections::HashMap;
use std::io;
use std::path::{Component, Path, PathBuf};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock, PoisonError, Weak};

const REGION_SIZE: i32 = 32;

type RegionRegistry = HashMap<PathBuf, Weak<SharedRegionFile>>;

// Paths are normalized before lookup. Publishing an empty entry under this
// short-held registry lock lets the entry's own mutex serialize lazy open and
// file I/O without serializing unrelated region files.
static REGION_REGISTRY: OnceLock<Mutex<RegionRegistry>> = OnceLock::new();

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RegionKey {
    lod_index: u8,
    region_position: Vector3i,
}

impl RegionKey {
    fn from_block_position(position: Vector3i, lod_index: u8, region_size: i32) -> Self {
        let region_size = region_size.max(1);
        Self {
            lod_index,
            region_position: Vector3i::new(
                position.x.div_euclid(region_size),
                position.y.div_euclid(region_size),
                position.z.div_euclid(region_size),
            ),
        }
    }

    fn local_block_position(self, position: Vector3i, region_size: i32) -> Vector3i {
        let region_size = region_size.max(1);
        Vector3i::new(
            position.x.rem_euclid(region_size),
            position.y.rem_euclid(region_size),
            position.z.rem_euclid(region_size),
        )
    }
}

struct SharedRegionFile {
    path: PathBuf,
    file: Mutex<Option<RegionFile>>,
}

impl SharedRegionFile {
    fn new(path: PathBuf) -> Self {
        Self {
            path,
            file: Mutex::new(None),
        }
    }

    fn open(
        &self,
        create_if_not_found: bool,
        format: RegionFormat,
    ) -> Result<bool, VoxelStreamError> {
        let mut file = self.lock();
        if file.is_some() {
            return Ok(true);
        }

        match RegionFile::open_with_format(&self.path, create_if_not_found, format) {
            Ok(region) => {
                *file = Some(region);
                Ok(true)
            }
            Err(RegionError::NotFound(_)) if !create_if_not_found => Ok(false),
            Err(error) => Err(map_region_error(error, &self.path)),
        }
    }

    fn lock(&self) -> MutexGuard<'_, Option<RegionFile>> {
        self.file.lock().unwrap_or_else(PoisonError::into_inner)
    }

    fn with_file<T>(
        &self,
        operation: impl FnOnce(&mut RegionFile) -> Result<T, RegionError>,
    ) -> Result<T, RegionError> {
        let mut file = self.lock();
        let Some(file) = file.as_mut() else {
            return Err(RegionError::Io(
                "shared region file was used before it was opened".to_owned(),
            ));
        };
        operation(file)
    }
}

struct MetaState {
    loaded: bool,
    saved: bool,
    meta: RegionForestMeta,
}

pub struct RegionFilesStream {
    directory: PathBuf,
    region_size: i32,
    sector_size: u32,
    block_size_po2: u8,
    meta: Mutex<MetaState>,
    regions: Mutex<HashMap<RegionKey, Arc<SharedRegionFile>>>,
}

impl RegionFilesStream {
    pub fn new(directory: PathBuf) -> Self {
        Self::with_settings(directory, REGION_SIZE, 512)
    }

    /// Construct a stream with inspector-configured region/sector sizes.
    /// `region_size` is blocks per axis (clamped to `1..=255` to match the
    /// on-disk `RegionFormat` byte field). `sector_size` is bytes.
    pub fn with_settings(directory: PathBuf, region_size: i32, sector_size: u32) -> Self {
        Self::with_block_size(directory, region_size, sector_size, 4)
    }

    /// Same as [`with_settings`](Self::with_settings) plus the forest
    /// `block_size_po2` written to `meta.vxrm`.
    pub fn with_block_size(
        directory: PathBuf,
        region_size: i32,
        sector_size: u32,
        block_size_po2: u8,
    ) -> Self {
        let region_size = region_size.clamp(1, 255);
        let sector_size = sector_size.max(1);
        let block_size_po2 = block_size_po2.clamp(1, 8);
        Self {
            directory: normalize_directory(directory),
            region_size,
            sector_size,
            block_size_po2,
            meta: Mutex::new(MetaState {
                loaded: false,
                saved: false,
                meta: RegionForestMeta::from_settings(block_size_po2, region_size, sector_size),
            }),
            regions: Mutex::new(HashMap::new()),
        }
    }

    pub fn region_size(&self) -> i32 {
        self.effective_region_size()
    }

    pub fn sector_size(&self) -> u32 {
        self.meta
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .meta
            .sector_size
    }

    pub fn block_size_po2(&self) -> u8 {
        self.meta
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .meta
            .block_size_po2
    }

    pub fn channel_depths(&self) -> [crate::storage::ChannelDepth; MAX_CHANNELS] {
        self.meta
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .meta
            .channel_depths
    }

    /// Rewrite every `.vxr` under `source` into `destination` using
    /// `new_region_size` / `new_sector_size`. Returns the number of blocks
    /// copied. Source files are left untouched.
    pub fn convert_directory(
        source: PathBuf,
        destination: PathBuf,
        new_region_size: i32,
        new_sector_size: u32,
    ) -> Result<u32, VoxelStreamError> {
        Self::convert_directory_ex(source, destination, new_region_size, new_sector_size, 4)
    }

    /// Rewrite region files and `meta.vxrm` with an explicit `block_size_po2`.
    pub fn convert_directory_ex(
        source: PathBuf,
        destination: PathBuf,
        new_region_size: i32,
        new_sector_size: u32,
        new_block_size_po2: u8,
    ) -> Result<u32, VoxelStreamError> {
        let dest = RegionFilesStream::with_block_size(
            destination,
            new_region_size,
            new_sector_size,
            new_block_size_po2,
        );
        let mut copied = 0u32;
        visit_region_files(&source, |path, lod_index, region_position| {
            let mut file = RegionFile::open(&path, false).map_err(|error| {
                VoxelStreamError::Io(format!("open {}: {error}", path.display()))
            })?;
            let region_size = file.format().region_size.x.max(1);
            let count = file.header_block_count();
            for index in 0..count {
                let local = file.block_position_from_index(index as u32);
                if !file.has_block(local) {
                    continue;
                }
                let mut buffer = VoxelBuffer::with_size(Vector3i::splat(1));
                match file.load_block(local, &mut buffer) {
                    Ok(()) => {}
                    Err(RegionError::BlockNotFound | RegionError::NotFound(_)) => continue,
                    Err(error) => {
                        return Err(map_region_error(error, &path));
                    }
                }
                let world = Vector3i::new(
                    region_position.x.saturating_mul(region_size) + local.x,
                    region_position.y.saturating_mul(region_size) + local.y,
                    region_position.z.saturating_mul(region_size) + local.z,
                );
                dest.save_voxel_block(VoxelSaveQuery::new(&buffer, world, lod_index))?;
                copied = copied.saturating_add(1);
            }
            Ok(())
        })?;
        dest.flush()?;
        let mut meta = dest.current_meta();
        meta.region_size_po2 = region_size_po2(new_region_size);
        meta.sector_size = new_sector_size;
        if copied == 0 {
            if let Ok(Some(source_meta)) = RegionForestMeta::load(&source) {
                meta.channel_depths = source_meta.channel_depths;
                meta.block_size_po2 = source_meta.block_size_po2;
            }
        }
        meta.save(&dest.directory)
            .map_err(|error| VoxelStreamError::Io(error.to_string()))?;
        Ok(copied)
    }

    fn region_path(&self, key: RegionKey) -> PathBuf {
        let position = key.region_position;
        self.directory
            .join(format!("lod{}", key.lod_index))
            .join(format!(
                "r.{}.{}.{}.vxr",
                position.x, position.y, position.z
            ))
    }

    fn legacy_region_path(&self, key: RegionKey) -> Option<PathBuf> {
        if key.lod_index != 0 {
            return None;
        }
        let position = key.region_position;
        Some(self.directory.join(format!(
            "r.{}.{}.{}.vxr",
            position.x, position.y, position.z
        )))
    }

    fn get_cached_region(&self, key: RegionKey) -> Option<Arc<SharedRegionFile>> {
        self.lock_regions().get(&key).cloned()
    }

    fn get_or_open_region(
        &self,
        key: RegionKey,
        create_if_not_found: bool,
        format: RegionFormat,
    ) -> Result<Option<Arc<SharedRegionFile>>, VoxelStreamError> {
        if let Some(region) = self.get_cached_region(key) {
            return Ok(Some(region));
        }

        let current_path = self.region_path(key);
        // The current layout is authoritative when both layouts exist.
        if let Some(region) = open_shared_region(current_path.clone(), false, format.clone())? {
            return Ok(Some(self.cache_region(key, region)));
        }

        // Keep an existing legacy LOD0 file as this region's read/write target.
        // Creating the current path only after both probes prevents split data.
        if let Some(legacy_path) = self.legacy_region_path(key) {
            if let Some(region) = open_shared_region(legacy_path, false, format.clone())? {
                return Ok(Some(self.cache_region(key, region)));
            }
        }

        if !create_if_not_found {
            return Ok(None);
        }

        let lod_directory = current_path
            .parent()
            .expect("region path always has a LOD directory");
        std::fs::create_dir_all(lod_directory).map_err(|error| {
            VoxelStreamError::Io(format!(
                "create region directory {}: {error}",
                lod_directory.display()
            ))
        })?;
        let region = open_shared_region(current_path, true, format)?
            .expect("create_if_not_found always returns a region or an error");
        Ok(Some(self.cache_region(key, region)))
    }

    fn cache_region(&self, key: RegionKey, region: Arc<SharedRegionFile>) -> Arc<SharedRegionFile> {
        self.lock_regions().entry(key).or_insert(region).clone()
    }

    fn lock_regions(&self) -> MutexGuard<'_, HashMap<RegionKey, Arc<SharedRegionFile>>> {
        self.regions.lock().unwrap_or_else(PoisonError::into_inner)
    }

    fn lock_meta(&self) -> MutexGuard<'_, MetaState> {
        self.meta.lock().unwrap_or_else(PoisonError::into_inner)
    }

    fn effective_region_size(&self) -> i32 {
        let state = self.lock_meta();
        if state.loaded {
            state.meta.region_size_blocks()
        } else {
            self.region_size
        }
    }

    fn ensure_meta_loaded(&self) -> Result<bool, VoxelStreamError> {
        let mut state = self.lock_meta();
        if state.loaded {
            return Ok(state.saved);
        }
        match RegionForestMeta::load(&self.directory) {
            Ok(Some(meta)) => {
                state.meta = meta;
                state.loaded = true;
                state.saved = true;
                Ok(true)
            }
            Ok(None) | Err(super::forest_meta::ForestMetaError::Io(_)) => {
                // Missing or unreadable sidecar: keep going so region-file I/O
                // can report the concrete path error (symlink loops, ENOTDIR).
                state.loaded = true;
                state.saved = false;
                Ok(false)
            }
            Err(super::forest_meta::ForestMetaError::Invalid(message)) => {
                Err(VoxelStreamError::CorruptData(format!(
                    "{}: {message}",
                    self.directory.join(META_FILE_NAME).display()
                )))
            }
        }
    }

    fn persist_meta_from_block(
        &self,
        buffer: &VoxelBuffer,
    ) -> Result<RegionForestMeta, VoxelStreamError> {
        let _ = self.ensure_meta_loaded()?;
        // First save: derive the candidate from this stream's settings and
        // the buffer, and validate BEFORE touching the filesystem. Writing
        // meta.vxrm for a mismatched buffer (e.g. a wrong inspector
        // `block_size_po2`) would lock the forest into a format no block can
        // ever match, poisoning every future save while no voxels exist on
        // disk yet.
        let candidate = {
            let state = self.lock_meta();
            if state.saved {
                if state.meta.matches_buffer(buffer) {
                    return Ok(state.meta.clone());
                }
                return Err(VoxelStreamError::BlockFormatMismatch);
            }
            let mut candidate = state.meta.clone();
            candidate.block_size_po2 = self.block_size_po2;
            candidate.region_size_po2 = region_size_po2(self.region_size);
            candidate.sector_size = self.sector_size;
            candidate.capture_channel_depths(buffer);
            if !candidate.matches_buffer(buffer) {
                return Err(VoxelStreamError::BlockFormatMismatch);
            }
            candidate
        };
        // Write outside the meta lock: filesystem I/O must not hold up
        // concurrent meta readers.
        candidate
            .save(&self.directory)
            .map_err(|error| VoxelStreamError::Io(error.to_string()))?;
        let mut state = self.lock_meta();
        if !state.saved {
            state.meta = candidate;
            state.saved = true;
            state.loaded = true;
        } else if !state.meta.matches_buffer(buffer) {
            // Lost the first-save race: this thread's candidate bytes may be
            // the ones on disk while the winner's format is authoritative.
            // Restore the committed meta so the file cannot poison the next
            // session with a format no block matches.
            state
                .meta
                .save(&self.directory)
                .map_err(|error| VoxelStreamError::Io(error.to_string()))?;
            return Err(VoxelStreamError::BlockFormatMismatch);
        }
        Ok(state.meta.clone())
    }

    fn current_meta(&self) -> RegionForestMeta {
        self.lock_meta().meta.clone()
    }
}

impl VoxelStream for RegionFilesStream {
    fn load_voxel_block(&self, query: VoxelLoadQuery<'_>) -> StreamResult<LoadResult> {
        let has_meta = self.ensure_meta_loaded()?;
        let meta = self.current_meta();
        if has_meta {
            meta.apply_channel_depths(query.voxel_buffer);
        }
        let region_size = meta.region_size_blocks();
        let key =
            RegionKey::from_block_position(query.position_in_blocks, query.lod_index, region_size);
        let local_position = key.local_block_position(query.position_in_blocks, region_size);
        let format = if has_meta {
            meta.to_region_format()
        } else {
            RegionFormat::default()
        };
        let region = if let Some(region) = self.get_cached_region(key) {
            region
        } else {
            let Some(region) = self.get_or_open_region(key, false, format)? else {
                return Ok(LoadResult::NotFound);
            };
            region
        };
        let result = region.with_file(|file| file.load_block(local_position, query.voxel_buffer));
        match result {
            Ok(()) => Ok(LoadResult::Found),
            Err(RegionError::BlockNotFound | RegionError::NotFound(_)) => Ok(LoadResult::NotFound),
            Err(error) => Err(map_region_error(error, &region.path)),
        }
    }

    fn save_voxel_block(&self, query: VoxelSaveQuery<'_>) -> StreamResult<()> {
        let meta = self.persist_meta_from_block(query.voxel_buffer)?;
        let region_size = meta.region_size_blocks();
        let key =
            RegionKey::from_block_position(query.position_in_blocks, query.lod_index, region_size);
        let local_position = key.local_block_position(query.position_in_blocks, region_size);
        let region = self
            .get_or_open_region(key, true, meta.to_region_format())?
            .expect("create_if_not_found always returns a region or an error");
        region
            .with_file(|file| file.save_block(local_position, query.voxel_buffer, Compression::Lz4))
            .map_err(|error| map_region_error(error, &region.path))
    }

    fn get_used_channels_mask(&self) -> u8 {
        ALL_CHANNELS_MASK
    }

    fn get_lod_count(&self) -> u8 {
        MAX_LOD as u8
    }

    fn get_supported_save_mode(&self) -> SaveMode {
        SaveMode::Filesystem
    }

    fn flush(&self) -> StreamResult<()> {
        let regions = self.lock_regions().values().cloned().collect::<Vec<_>>();
        flush_all(regions, |region| {
            region
                .with_file(RegionFile::flush)
                .map_err(|error| map_region_error(error, &region.path))
        })
    }
}

fn open_shared_region(
    path: PathBuf,
    create_if_not_found: bool,
    format: RegionFormat,
) -> Result<Option<Arc<SharedRegionFile>>, VoxelStreamError> {
    // Never publish a lexical fallback: another alias may become resolvable
    // before this thread reaches `RegionFile::open`, yielding two handles.
    let Some(path) = region_identity_path(&path).map_err(|error| {
        VoxelStreamError::Io(format!("resolve region path {}: {error}", path.display()))
    })?
    else {
        if create_if_not_found {
            return Err(VoxelStreamError::Io(format!(
                "resolve region path {} after creating its parent",
                path.display()
            )));
        }
        return Ok(None);
    };
    let region = shared_region(path);
    if region.open(create_if_not_found, format)? {
        Ok(Some(region))
    } else {
        Ok(None)
    }
}

fn region_identity_path(path: &Path) -> io::Result<Option<PathBuf>> {
    region_identity_path_with(
        path,
        |path| std::fs::canonicalize(path),
        |path| std::fs::metadata(path).map(|metadata| metadata.is_dir()),
    )
}

fn region_identity_path_with<C, D>(
    path: &Path,
    mut canonicalize: C,
    mut is_directory: D,
) -> io::Result<Option<PathBuf>>
where
    C: FnMut(&Path) -> io::Result<PathBuf>,
    D: FnMut(&Path) -> io::Result<bool>,
{
    match canonicalize(path) {
        Ok(canonical) => return Ok(Some(canonical)),
        Err(error) if is_resolution_absence(error.kind()) => {}
        Err(error) => return Err(error),
    }

    let Some((parent, file_name)) = path.parent().zip(path.file_name()) else {
        return Ok(None);
    };
    let canonical_parent = match canonicalize(parent) {
        Ok(canonical_parent) => canonical_parent,
        Err(error) if is_resolution_absence(error.kind()) => return Ok(None),
        Err(error) => return Err(error),
    };
    match is_directory(&canonical_parent) {
        Ok(true) => Ok(Some(canonical_parent.join(file_name))),
        Ok(false) => Ok(None),
        Err(error) if is_resolution_absence(error.kind()) => Ok(None),
        Err(error) => Err(error),
    }
}

fn is_resolution_absence(kind: io::ErrorKind) -> bool {
    matches!(kind, io::ErrorKind::NotFound | io::ErrorKind::NotADirectory)
}

fn shared_region(path: PathBuf) -> Arc<SharedRegionFile> {
    let registry = REGION_REGISTRY.get_or_init(|| Mutex::new(HashMap::new()));
    let mut registry = registry.lock().unwrap_or_else(PoisonError::into_inner);
    if let Some(region) = registry.get(&path).and_then(Weak::upgrade) {
        return region;
    }

    registry.retain(|_, region| region.strong_count() > 0);
    let region = Arc::new(SharedRegionFile::new(path.clone()));
    registry.insert(path, Arc::downgrade(&region));
    region
}

fn normalize_directory(directory: PathBuf) -> PathBuf {
    let absolute = if directory.is_absolute() {
        directory
    } else {
        match std::env::current_dir() {
            Ok(current_directory) => current_directory.join(directory),
            Err(_) => directory,
        }
    };

    let mut ancestor = Some(absolute.as_path());
    while let Some(path) = ancestor {
        if let Ok(canonical) = path.canonicalize() {
            let suffix = absolute.strip_prefix(path).unwrap_or(Path::new(""));
            return normalize_lexically(canonical.join(suffix));
        }
        ancestor = path.parent();
    }
    normalize_lexically(absolute)
}

fn normalize_lexically(path: PathBuf) -> PathBuf {
    let is_absolute = path.is_absolute();
    let mut normalized = PathBuf::new();
    for component in path.components() {
        match component {
            Component::Prefix(_) | Component::RootDir | Component::Normal(_) => {
                normalized.push(component.as_os_str());
            }
            Component::CurDir => {}
            Component::ParentDir => {
                if !normalized.pop() && !is_absolute {
                    normalized.push(component.as_os_str());
                }
            }
        }
    }
    normalized
}

fn flush_all<I, T, F, E>(items: I, mut flush: F) -> Result<(), E>
where
    I: IntoIterator<Item = T>,
    F: FnMut(T) -> Result<(), E>,
{
    let mut first_error = None;
    for item in items {
        if let Err(error) = flush(item) {
            first_error.get_or_insert(error);
        }
    }
    match first_error {
        Some(error) => Err(error),
        None => Ok(()),
    }
}

fn visit_region_files(
    root: &Path,
    mut visit: impl FnMut(PathBuf, u8, Vector3i) -> Result<(), VoxelStreamError>,
) -> Result<(), VoxelStreamError> {
    if !root.is_dir() {
        return Ok(());
    }
    visit_region_dir(root, 0, &mut visit)?;
    for lod in 0..=MAX_LOD as u8 {
        let lod_dir = root.join(format!("lod{lod}"));
        if lod_dir.is_dir() {
            visit_region_dir(&lod_dir, lod, &mut visit)?;
        }
    }
    Ok(())
}

fn visit_region_dir(
    dir: &Path,
    lod_index: u8,
    visit: &mut impl FnMut(PathBuf, u8, Vector3i) -> Result<(), VoxelStreamError>,
) -> Result<(), VoxelStreamError> {
    let entries = std::fs::read_dir(dir).map_err(|error| {
        VoxelStreamError::Io(format!("read region directory {}: {error}", dir.display()))
    })?;
    for entry in entries {
        let entry = entry.map_err(|error| {
            VoxelStreamError::Io(format!("read region entry {}: {error}", dir.display()))
        })?;
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
            continue;
        };
        let Some(position) = parse_region_file_name(name) else {
            continue;
        };
        visit(path, lod_index, position)?;
    }
    Ok(())
}

fn parse_region_file_name(name: &str) -> Option<Vector3i> {
    let name = name.strip_suffix(".vxr")?;
    let mut parts = name.split('.');
    if parts.next()? != "r" {
        return None;
    }
    let x = parts.next()?.parse().ok()?;
    let y = parts.next()?.parse().ok()?;
    let z = parts.next()?.parse().ok()?;
    if parts.next().is_some() {
        return None;
    }
    Some(Vector3i::new(x, y, z))
}

#[cfg_attr(not(test), allow(dead_code))]
fn format_for_block(
    block: &VoxelBuffer,
    region_size: i32,
    sector_size: u32,
) -> Result<RegionFormat, VoxelStreamError> {
    let size = block.size();
    if size.x <= 0
        || size.x != size.y
        || size.x != size.z
        || !u32::try_from(size.x).is_ok_and(|axis_size| axis_size.is_power_of_two())
    {
        return Err(VoxelStreamError::BlockFormatMismatch);
    }

    let block_size_po2 =
        u8::try_from(size.x.ilog2()).map_err(|_| VoxelStreamError::BlockFormatMismatch)?;
    let mut format = RegionFormat {
        block_size_po2,
        region_size: Vector3i::splat(region_size),
        sector_size,
        ..RegionFormat::default()
    };
    for channel_index in 0..MAX_CHANNELS {
        format.channel_depths[channel_index] = block.channel_depth(channel_index);
    }
    format
        .validate_result()
        .map_err(|_| VoxelStreamError::BlockFormatMismatch)?;
    Ok(format)
}

fn map_region_error(error: RegionError, path: &Path) -> VoxelStreamError {
    match error {
        RegionError::BlockFormatMismatch => VoxelStreamError::BlockFormatMismatch,
        RegionError::InvalidBlockPosition => VoxelStreamError::InvalidBlockPosition {
            position: Vector3i::zero(),
        },
        RegionError::NotFound(message) | RegionError::Io(message) => {
            VoxelStreamError::Io(format!("region {}: {message}", path.display()))
        }
        RegionError::BadHeader(message) => {
            VoxelStreamError::CorruptData(format!("region {}: {message}", path.display()))
        }
        RegionError::UnsupportedVersion(version) => VoxelStreamError::CorruptData(format!(
            "region {} has unsupported version {version}",
            path.display()
        )),
        RegionError::BlockSerializer(error) => {
            VoxelStreamError::CorruptData(format!("region {} block data: {error}", path.display()))
        }
        RegionError::BlockNotFound => VoxelStreamError::CorruptData(format!(
            "region {} block unexpectedly missing",
            path.display()
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::{
        flush_all, format_for_block, open_shared_region, region_identity_path_with,
        RegionFilesStream, RegionKey, REGION_REGISTRY, REGION_SIZE,
    };
    use crate::math::Vector3i;
    use crate::storage::{Allocator, ChannelId, VoxelBuffer};
    use crate::streams::compressed_data::Compression;
    use crate::streams::region::forest_meta::{RegionForestMeta, META_FILE_NAME};
    use crate::streams::region::RegionFile;
    use crate::streams::{
        LoadResult, VoxelLoadQuery, VoxelSaveQuery, VoxelStream, VoxelStreamError,
    };
    use std::io;
    use std::path::{Path, PathBuf};
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::{mpsc, Arc, Barrier};
    use std::time::Duration;

    static NEXT_TEMP_DIR: AtomicU64 = AtomicU64::new(0);

    struct TestDir(PathBuf);

    impl TestDir {
        fn new() -> Self {
            let unique = NEXT_TEMP_DIR.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!(
                "godot-voxel-region-stream-{}-{unique}",
                std::process::id()
            ));
            std::fs::create_dir_all(&path).unwrap();
            Self(path)
        }

        fn path(&self) -> &Path {
            &self.0
        }
    }

    impl Drop for TestDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn sample_block(value: u64) -> VoxelBuffer {
        let mut block = VoxelBuffer::with_size(Vector3i::splat(16));
        block.set_voxel(value, 1, 2, 3, ChannelId::Type.index());
        block
    }

    fn save(
        stream: &RegionFilesStream,
        position: Vector3i,
        lod_index: u8,
        block: &VoxelBuffer,
    ) -> Result<(), VoxelStreamError> {
        stream.save_voxel_block(VoxelSaveQuery::new(block, position, lod_index))
    }

    fn load(
        stream: &RegionFilesStream,
        position: Vector3i,
        lod_index: u8,
    ) -> Result<(LoadResult, VoxelBuffer), VoxelStreamError> {
        let mut block = VoxelBuffer::new(Allocator::Default);
        let result =
            stream.load_voxel_block(VoxelLoadQuery::new(&mut block, position, lod_index))?;
        Ok((result, block))
    }

    #[test]
    fn round_trips_negative_block_position() {
        let dir = TestDir::new();
        let stream = RegionFilesStream::new(dir.path().to_path_buf());
        let position = Vector3i::new(-1, -33, 2);

        save(&stream, position, 0, &sample_block(41)).unwrap();
        stream.flush().unwrap();
        drop(stream);
        let reopened = RegionFilesStream::new(dir.path().to_path_buf());
        let (result, loaded) = load(&reopened, position, 0).unwrap();

        assert_eq!(result, LoadResult::Found);
        assert_eq!(loaded.get_voxel(1, 2, 3, ChannelId::Type.index()), 41);
        assert!(dir.path().join("lod0/r.-1.-2.0.vxr").is_file());
    }

    #[test]
    fn with_settings_uses_configured_region_size() {
        let dir = TestDir::new();
        let stream = RegionFilesStream::with_settings(dir.path().to_path_buf(), 16, 256);
        assert_eq!(stream.region_size(), 16);
        assert_eq!(stream.sector_size(), 256);
        save(&stream, Vector3i::new(17, 0, 0), 0, &sample_block(9)).unwrap();
        stream.flush().unwrap();
        drop(stream);
        let reopened = RegionFilesStream::with_settings(dir.path().to_path_buf(), 16, 256);
        let (result, loaded) = load(&reopened, Vector3i::new(17, 0, 0), 0).unwrap();
        assert_eq!(result, LoadResult::Found);
        assert_eq!(loaded.get_voxel(1, 2, 3, ChannelId::Type.index()), 9);
        // region_size 16 → block 17 maps to region x=1.
        assert!(dir.path().join("lod0/r.1.0.0.vxr").is_file());
    }

    #[test]
    fn convert_directory_rewrites_blocks_under_new_region_size() {
        let src = TestDir::new();
        let stream = RegionFilesStream::with_settings(src.path().to_path_buf(), 32, 512);
        save(&stream, Vector3i::new(17, 0, 0), 0, &sample_block(11)).unwrap();
        stream.flush().unwrap();
        drop(stream);

        let dest = TestDir::new();
        let copied = RegionFilesStream::convert_directory(
            src.path().to_path_buf(),
            dest.path().to_path_buf(),
            16,
            256,
        )
        .unwrap();
        assert_eq!(copied, 1);
        let reopened = RegionFilesStream::with_settings(dest.path().to_path_buf(), 16, 256);
        let (result, loaded) = load(&reopened, Vector3i::new(17, 0, 0), 0).unwrap();
        assert_eq!(result, LoadResult::Found);
        assert_eq!(loaded.get_voxel(1, 2, 3, ChannelId::Type.index()), 11);
        assert!(dest.path().join("lod0/r.1.0.0.vxr").is_file());
        assert!(dest.path().join(META_FILE_NAME).is_file());
    }

    #[test]
    fn region_stream_round_trips_metadata_across_reopen() {
        let dir = TestDir::new();
        let stream = RegionFilesStream::new(dir.path().to_path_buf());
        let position = Vector3i::new(2, -3, 4);

        let mut block = sample_block(41);
        block.set_block_metadata(crate::storage::MetadataValue::Bytes(vec![9, 8, 7]));
        block.set_voxel_metadata(
            Vector3i::new(1, 2, 3),
            crate::storage::MetadataValue::Text("maple".into()),
        );
        save(&stream, position, 0, &block).unwrap();
        stream.flush().unwrap();
        drop(stream);

        let reopened = RegionFilesStream::new(dir.path().to_path_buf());
        let (result, loaded) = load(&reopened, position, 0).unwrap();
        assert_eq!(result, LoadResult::Found);
        assert_eq!(
            *loaded.block_metadata(),
            crate::storage::MetadataValue::Bytes(vec![9, 8, 7])
        );
        assert_eq!(
            loaded.voxel_metadata(Vector3i::new(1, 2, 3)),
            Some(&crate::storage::MetadataValue::Text("maple".into()))
        );
        assert_eq!(loaded.get_voxel(1, 2, 3, ChannelId::Type.index()), 41);
    }

    #[test]
    fn convert_directory_preserves_metadata() {
        let src = TestDir::new();
        let stream = RegionFilesStream::with_settings(src.path().to_path_buf(), 32, 512);
        let mut block = sample_block(11);
        block.set_block_metadata(crate::storage::MetadataValue::Int(5));
        block.set_voxel_metadata(
            Vector3i::new(1, 2, 3),
            crate::storage::MetadataValue::Float(1.5),
        );
        save(&stream, Vector3i::new(17, 0, 0), 0, &block).unwrap();
        stream.flush().unwrap();
        drop(stream);

        let dest = TestDir::new();
        let copied = RegionFilesStream::convert_directory(
            src.path().to_path_buf(),
            dest.path().to_path_buf(),
            16,
            256,
        )
        .unwrap();
        assert_eq!(copied, 1);
        let reopened = RegionFilesStream::with_settings(dest.path().to_path_buf(), 16, 256);
        let (result, loaded) = load(&reopened, Vector3i::new(17, 0, 0), 0).unwrap();
        assert_eq!(result, LoadResult::Found);
        assert_eq!(
            *loaded.block_metadata(),
            crate::storage::MetadataValue::Int(5)
        );
        assert_eq!(
            loaded.voxel_metadata(Vector3i::new(1, 2, 3)),
            Some(&crate::storage::MetadataValue::Float(1.5))
        );
    }

    #[test]
    fn first_save_with_mismatched_block_size_does_not_poison_meta() {
        // The stream's block_size_po2 (4 -> 16) does not match the actual
        // buffer (32): the first save must fail WITHOUT writing meta.vxrm,
        // so a subsequent correctly-sized save still succeeds.
        let dir = TestDir::new();
        let stream = RegionFilesStream::with_block_size(dir.path().to_path_buf(), 16, 256, 4);
        let wrong = VoxelBuffer::with_size(Vector3i::splat(32));

        assert_eq!(
            save(&stream, Vector3i::zero(), 0, &wrong),
            Err(VoxelStreamError::BlockFormatMismatch)
        );
        assert!(
            !dir.path().join(META_FILE_NAME).is_file(),
            "a rejected first save must not lock the forest format"
        );

        let right = VoxelBuffer::with_size(Vector3i::splat(16));
        save(&stream, Vector3i::zero(), 0, &right).unwrap();
        assert!(dir.path().join(META_FILE_NAME).is_file());
    }

    #[test]
    fn first_save_writes_meta_vxrm_and_locks_channel_depths() {
        let dir = TestDir::new();
        let stream = RegionFilesStream::with_block_size(dir.path().to_path_buf(), 16, 256, 4);
        let mut block = sample_block(3);
        block.set_channel_depth(ChannelId::Sdf.index(), crate::storage::ChannelDepth::Bit32);
        save(&stream, Vector3i::zero(), 0, &block).unwrap();
        stream.flush().unwrap();
        drop(stream);

        let meta_path = dir.path().join(META_FILE_NAME);
        assert!(meta_path.is_file());
        let meta =
            RegionForestMeta::from_json(&std::fs::read_to_string(meta_path).unwrap()).unwrap();
        assert_eq!(meta.block_size_po2, 4);
        assert_eq!(meta.region_size_po2, 4);
        assert_eq!(meta.sector_size, 256);
        assert_eq!(
            meta.channel_depths[ChannelId::Sdf.index()],
            crate::storage::ChannelDepth::Bit32
        );

        let reopened = RegionFilesStream::with_block_size(dir.path().to_path_buf(), 16, 256, 4);
        let (result, loaded) = load(&reopened, Vector3i::zero(), 0).unwrap();
        assert_eq!(result, LoadResult::Found);
        assert_eq!(
            loaded.channel_depth(ChannelId::Sdf.index()),
            crate::storage::ChannelDepth::Bit32
        );
        assert_eq!(loaded.get_voxel(1, 2, 3, ChannelId::Type.index()), 3);

        let mut other = sample_block(4);
        other.set_channel_depth(ChannelId::Sdf.index(), crate::storage::ChannelDepth::Bit8);
        assert_eq!(
            save(&reopened, Vector3i::new(1, 0, 0), 0, &other),
            Err(VoxelStreamError::BlockFormatMismatch)
        );
    }

    #[test]
    fn keeps_same_position_separate_across_lods() {
        let dir = TestDir::new();
        let stream = RegionFilesStream::new(dir.path().to_path_buf());
        let position = Vector3i::new(3, 4, 5);

        save(&stream, position, 0, &sample_block(10)).unwrap();
        save(&stream, position, 1, &sample_block(20)).unwrap();

        let (lod0_result, lod0) = load(&stream, position, 0).unwrap();
        let (lod1_result, lod1) = load(&stream, position, 1).unwrap();
        assert_eq!(lod0_result, LoadResult::Found);
        assert_eq!(lod1_result, LoadResult::Found);
        assert_eq!(lod0.get_voxel(1, 2, 3, ChannelId::Type.index()), 10);
        assert_eq!(lod1.get_voxel(1, 2, 3, ChannelId::Type.index()), 20);
    }

    #[test]
    fn concurrent_saves_to_one_region_preserve_every_block() {
        const BLOCK_COUNT: usize = 8;

        let dir = TestDir::new();
        let stream = Arc::new(RegionFilesStream::new(dir.path().to_path_buf()));
        let start = Arc::new(Barrier::new(BLOCK_COUNT));
        let mut threads = Vec::new();
        for index in 0..BLOCK_COUNT {
            let stream = stream.clone();
            let start = start.clone();
            threads.push(std::thread::spawn(move || {
                let block = sample_block(index as u64 + 1);
                start.wait();
                save(&stream, Vector3i::new(index as i32, 0, 0), 0, &block).unwrap();
            }));
        }
        for thread in threads {
            thread.join().unwrap();
        }
        stream.flush().unwrap();
        drop(stream);
        let reopened = RegionFilesStream::new(dir.path().to_path_buf());

        for index in 0..BLOCK_COUNT {
            let (result, block) = load(&reopened, Vector3i::new(index as i32, 0, 0), 0).unwrap();
            assert_eq!(result, LoadResult::Found);
            assert_eq!(
                block.get_voxel(1, 2, 3, ChannelId::Type.index()),
                index as u64 + 1
            );
        }
    }

    #[test]
    fn unresolved_load_probe_does_not_publish_a_registry_entry() {
        let dir = TestDir::new();
        let unresolved_path = dir
            .path()
            .join("not-created")
            .join("lod0")
            .join("r.0.0.0.vxr");

        let region = open_shared_region(
            unresolved_path.clone(),
            false,
            format_for_block(&sample_block(1), REGION_SIZE, 512).unwrap(),
        )
        .unwrap();

        assert!(region.is_none());
        if let Some(registry) = REGION_REGISTRY.get() {
            let registry = registry
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            assert!(
                !registry.contains_key(&unresolved_path),
                "an unresolved raw path was published before its physical identity existed"
            );
        }
    }

    #[test]
    fn missing_region_load_returns_not_found() {
        let dir = TestDir::new();
        let stream = RegionFilesStream::new(dir.path().join("not-created"));

        let (result, _) = load(&stream, Vector3i::zero(), 0).unwrap();

        assert_eq!(result, LoadResult::NotFound);
    }

    #[test]
    fn creating_region_requires_a_resolved_parent() {
        let dir = TestDir::new();
        let unresolved_path = dir
            .path()
            .join("not-created")
            .join("lod0")
            .join("r.0.0.0.vxr");

        let error = match open_shared_region(
            unresolved_path,
            true,
            format_for_block(&sample_block(1), REGION_SIZE, 512).unwrap(),
        ) {
            Err(error) => error,
            Ok(_) => panic!("creation must not publish a region without a resolved parent"),
        };

        assert!(matches!(error, VoxelStreamError::Io(_)));
    }

    #[test]
    fn resolver_classifies_missing_or_non_directory_parent_as_absent() {
        for parent_error in [io::ErrorKind::NotFound, io::ErrorKind::NotADirectory] {
            let path = Path::new("/unresolved/lod0/r.0.0.0.vxr");
            let resolved = region_identity_path_with(
                path,
                |candidate| {
                    if candidate == path {
                        Err(io::Error::new(io::ErrorKind::NotFound, "missing target"))
                    } else {
                        Err(io::Error::new(parent_error, "unusable parent"))
                    }
                },
                |_| panic!("metadata must not run for an unresolved parent"),
            )
            .unwrap();

            assert_eq!(resolved, None);
        }
    }

    #[test]
    fn resolver_propagates_permission_denied() {
        let error = region_identity_path_with(
            Path::new("/unreadable/lod0/r.0.0.0.vxr"),
            |_| {
                Err(io::Error::new(
                    io::ErrorKind::PermissionDenied,
                    "permission denied",
                ))
            },
            |_| panic!("metadata must not run after a canonicalization error"),
        )
        .unwrap_err();

        assert_eq!(error.kind(), io::ErrorKind::PermissionDenied);
    }

    #[test]
    fn resolver_propagates_parent_metadata_error() {
        let path = Path::new("/pending/lod0/r.0.0.0.vxr");
        let canonical_parent = PathBuf::from("/canonical/lod0");
        let error = region_identity_path_with(
            path,
            |candidate| {
                if candidate == path {
                    Err(io::Error::new(io::ErrorKind::NotFound, "missing target"))
                } else {
                    Ok(canonical_parent.clone())
                }
            },
            |_| Err(io::Error::new(io::ErrorKind::TimedOut, "metadata timeout")),
        )
        .unwrap_err();

        assert_eq!(error.kind(), io::ErrorKind::TimedOut);
    }

    #[cfg(unix)]
    #[test]
    fn region_identity_resolution_io_error_is_reported_not_missing() {
        let dir = TestDir::new();
        let directory = dir.path().join("loop");
        std::os::unix::fs::symlink("loop", &directory).unwrap();
        let stream = RegionFilesStream::new(directory);

        let error = load(&stream, Vector3i::zero(), 0).unwrap_err();

        let VoxelStreamError::Io(message) = error else {
            panic!("expected I/O error");
        };
        assert!(message.contains("resolve region path"));
        assert!(message.contains("r.0.0.0.vxr"));
    }

    #[test]
    fn separate_streams_share_one_handle_for_concurrent_saves() {
        let dir = TestDir::new();
        let directory = dir.path().join("not-created").join("voxel-data");
        let alias = dir
            .path()
            .join("not-created")
            .join("unused")
            .join("..")
            .join("voxel-data");
        let first = Arc::new(RegionFilesStream::new(directory.clone()));
        let second = Arc::new(RegionFilesStream::new(alias));
        let key = RegionKey::from_block_position(Vector3i::zero(), 0, REGION_SIZE);
        let format = format_for_block(&sample_block(1), REGION_SIZE, 512).unwrap();

        save(&first, Vector3i::zero(), 0, &sample_block(1)).unwrap();
        let first_region = first
            .get_or_open_region(key, true, format.clone())
            .unwrap()
            .unwrap();
        let second_region = second
            .get_or_open_region(key, true, format)
            .unwrap()
            .unwrap();
        assert!(
            Arc::ptr_eq(&first_region, &second_region),
            "separate streams opened distinct handles for one normalized region path"
        );

        let start = Arc::new(Barrier::new(2));
        let first_thread = {
            let stream = first.clone();
            let start = start.clone();
            std::thread::spawn(move || {
                start.wait();
                save(&stream, Vector3i::new(1, 0, 0), 0, &sample_block(11)).unwrap();
            })
        };
        let second_thread = {
            let stream = second.clone();
            let start = start.clone();
            std::thread::spawn(move || {
                start.wait();
                save(&stream, Vector3i::new(2, 0, 0), 0, &sample_block(22)).unwrap();
            })
        };
        first_thread.join().unwrap();
        second_thread.join().unwrap();
        first.flush().unwrap();
        second.flush().unwrap();
        drop(first_region);
        drop(second_region);
        drop(first);
        drop(second);

        let reopened = RegionFilesStream::new(directory);
        for (position, expected) in [(Vector3i::new(1, 0, 0), 11), (Vector3i::new(2, 0, 0), 22)] {
            let (result, block) = load(&reopened, position, 0).unwrap();
            assert_eq!(result, LoadResult::Found);
            assert_eq!(block.get_voxel(1, 2, 3, ChannelId::Type.index()), expected);
        }
    }

    #[cfg(unix)]
    #[test]
    fn simultaneous_first_saves_through_physical_aliases_share_one_handle() {
        let dir = TestDir::new();
        let physical_directory = dir.path().join("physical");
        let first_directory = dir.path().join("first-alias").join("voxel-data");
        let second_directory = dir.path().join("second-alias").join("voxel-data");
        let first = Arc::new(RegionFilesStream::new(first_directory));
        let second = Arc::new(RegionFilesStream::new(second_directory));
        std::fs::create_dir_all(&physical_directory).unwrap();
        std::os::unix::fs::symlink(&physical_directory, dir.path().join("first-alias")).unwrap();
        std::os::unix::fs::symlink(&physical_directory, dir.path().join("second-alias")).unwrap();

        let key = RegionKey::from_block_position(Vector3i::zero(), 0, REGION_SIZE);
        let start = Arc::new(Barrier::new(2));
        let first_thread = {
            let stream = first.clone();
            let start = start.clone();
            std::thread::spawn(move || {
                start.wait();
                save(&stream, Vector3i::new(1, 0, 0), 0, &sample_block(11)).unwrap();
            })
        };
        let second_thread = {
            let stream = second.clone();
            let start = start.clone();
            std::thread::spawn(move || {
                start.wait();
                save(&stream, Vector3i::new(2, 0, 0), 0, &sample_block(22)).unwrap();
            })
        };
        first_thread.join().unwrap();
        second_thread.join().unwrap();

        let first_region = first.get_cached_region(key).unwrap();
        let second_region = second.get_cached_region(key).unwrap();
        assert!(
            Arc::ptr_eq(&first_region, &second_region),
            "physical aliases opened distinct handles for one region file"
        );
        first.flush().unwrap();
        second.flush().unwrap();
        first.lock_regions().clear();
        second.lock_regions().clear();
        drop(first_region);
        drop(second_region);

        let reload_start = Arc::new(Barrier::new(2));
        let first_load = {
            let stream = first.clone();
            let start = reload_start.clone();
            std::thread::spawn(move || {
                start.wait();
                load(&stream, Vector3i::new(1, 0, 0), 0).unwrap()
            })
        };
        let second_load = {
            let stream = second.clone();
            let start = reload_start.clone();
            std::thread::spawn(move || {
                start.wait();
                load(&stream, Vector3i::new(2, 0, 0), 0).unwrap()
            })
        };
        let (first_result, first_block) = first_load.join().unwrap();
        let (second_result, second_block) = second_load.join().unwrap();
        assert_eq!(first_result, LoadResult::Found);
        assert_eq!(second_result, LoadResult::Found);
        assert_eq!(first_block.get_voxel(1, 2, 3, ChannelId::Type.index()), 11);
        assert_eq!(second_block.get_voxel(1, 2, 3, ChannelId::Type.index()), 22);
        let first_region = first.get_cached_region(key).unwrap();
        let second_region = second.get_cached_region(key).unwrap();
        assert!(
            Arc::ptr_eq(&first_region, &second_region),
            "existing physical aliases opened distinct handles for one region file"
        );
        drop(first_region);
        drop(second_region);
        drop(first);
        drop(second);

        let reopened = RegionFilesStream::new(physical_directory.join("voxel-data"));
        for (position, expected) in [(Vector3i::new(1, 0, 0), 11), (Vector3i::new(2, 0, 0), 22)] {
            let (result, block) = load(&reopened, position, 0).unwrap();
            assert_eq!(result, LoadResult::Found);
            assert_eq!(block.get_voxel(1, 2, 3, ChannelId::Type.index()), expected);
        }
    }

    #[test]
    fn current_lod0_region_is_authoritative_when_legacy_also_exists() {
        let dir = TestDir::new();
        let current_directory = dir.path().join("lod0");
        std::fs::create_dir_all(&current_directory).unwrap();
        let current_path = current_directory.join("r.0.0.0.vxr");
        let legacy_path = dir.path().join("r.0.0.0.vxr");
        let existing_position = Vector3i::new(3, 0, 0);
        let added_position = Vector3i::new(4, 0, 0);
        let format = format_for_block(&sample_block(1), REGION_SIZE, 512).unwrap();

        let mut legacy = RegionFile::open_with_format(&legacy_path, true, format.clone()).unwrap();
        legacy
            .save_block(existing_position, &sample_block(31), Compression::Lz4)
            .unwrap();
        legacy.flush().unwrap();
        drop(legacy);

        let mut current = RegionFile::open_with_format(&current_path, true, format).unwrap();
        current
            .save_block(existing_position, &sample_block(59), Compression::Lz4)
            .unwrap();
        current.flush().unwrap();
        drop(current);

        let stream = RegionFilesStream::new(dir.path().to_path_buf());
        let (result, loaded) = load(&stream, existing_position, 0).unwrap();
        assert_eq!(result, LoadResult::Found);
        assert_eq!(loaded.get_voxel(1, 2, 3, ChannelId::Type.index()), 59);
        save(&stream, added_position, 0, &sample_block(67)).unwrap();
        stream.flush().unwrap();
        drop(stream);

        let mut legacy = RegionFile::open(&legacy_path, false).unwrap();
        let mut legacy_block = VoxelBuffer::new(Allocator::Default);
        assert!(matches!(
            legacy.load_block(added_position, &mut legacy_block),
            Err(crate::streams::region::RegionError::BlockNotFound)
        ));
    }

    #[test]
    fn legacy_lod0_region_remains_the_read_write_target() {
        let dir = TestDir::new();
        let legacy_path = dir.path().join("r.0.0.0.vxr");
        let original_position = Vector3i::new(3, 0, 0);
        let added_position = Vector3i::new(4, 0, 0);
        let original_block = sample_block(31);
        let mut legacy = RegionFile::open_with_format(
            &legacy_path,
            true,
            format_for_block(&original_block, REGION_SIZE, 512).unwrap(),
        )
        .unwrap();
        legacy
            .save_block(original_position, &original_block, Compression::Lz4)
            .unwrap();
        legacy.flush().unwrap();
        drop(legacy);

        let stream = RegionFilesStream::new(dir.path().to_path_buf());
        let (result, loaded) = load(&stream, original_position, 0).unwrap();
        assert_eq!(result, LoadResult::Found);
        assert_eq!(loaded.get_voxel(1, 2, 3, ChannelId::Type.index()), 31);

        save(&stream, added_position, 0, &sample_block(47)).unwrap();
        stream.flush().unwrap();
        drop(stream);

        assert!(
            !dir.path().join("lod0/r.0.0.0.vxr").exists(),
            "writing a discovered legacy region must not create a second LOD0 file"
        );
        let reopened = RegionFilesStream::new(dir.path().to_path_buf());
        for (position, expected) in [(original_position, 31), (added_position, 47)] {
            let (result, block) = load(&reopened, position, 0).unwrap();
            assert_eq!(result, LoadResult::Found);
            assert_eq!(block.get_voxel(1, 2, 3, ChannelId::Type.index()), expected);
        }
    }

    #[test]
    fn locking_one_region_does_not_block_saving_another() {
        let dir = TestDir::new();
        std::fs::create_dir_all(dir.path().join("lod0")).unwrap();
        let stream = Arc::new(RegionFilesStream::new(dir.path().to_path_buf()));
        let first_key = RegionKey::from_block_position(Vector3i::zero(), 0, REGION_SIZE);
        let first_region = stream
            .get_or_open_region(
                first_key,
                true,
                format_for_block(&sample_block(1), REGION_SIZE, 512).unwrap(),
            )
            .unwrap()
            .unwrap();
        let first_guard = first_region.lock();
        let (sender, receiver) = mpsc::channel();
        let worker = {
            let stream = stream.clone();
            std::thread::spawn(move || {
                let result = save(&stream, Vector3i::new(32, 0, 0), 0, &sample_block(2));
                sender.send(result).unwrap();
            })
        };

        let result = receiver.recv_timeout(Duration::from_secs(2));
        drop(first_guard);
        worker.join().unwrap();

        assert!(
            matches!(result, Ok(Ok(()))),
            "a different region was blocked by an unrelated region lock: {result:?}"
        );
    }

    #[test]
    fn corrupt_region_is_reported_not_missing() {
        let dir = TestDir::new();
        let lod_dir = dir.path().join("lod0");
        std::fs::create_dir_all(&lod_dir).unwrap();
        std::fs::write(lod_dir.join("r.0.0.0.vxr"), b"not a region").unwrap();
        let stream = RegionFilesStream::new(dir.path().to_path_buf());

        let error = load(&stream, Vector3i::zero(), 0).unwrap_err();

        assert!(matches!(error, VoxelStreamError::CorruptData(_)));
    }

    #[test]
    fn directory_creation_failure_is_reported() {
        let dir = TestDir::new();
        let file_path = dir.path().join("not-a-directory");
        std::fs::write(&file_path, b"file").unwrap();
        let stream = RegionFilesStream::new(file_path);

        let error = save(&stream, Vector3i::zero(), 0, &sample_block(1)).unwrap_err();

        assert!(matches!(error, VoxelStreamError::Io(_)));
    }

    #[test]
    fn flush_flushes_all_cached_regions() {
        let mut attempted = Vec::new();

        let result = flush_all([0, 1, 2], |region| {
            attempted.push(region);
            if region == 1 {
                Err("middle region failed")
            } else {
                Ok(())
            }
        });

        assert_eq!(attempted, [0, 1, 2]);
        assert_eq!(result, Err("middle region failed"));
    }
}
