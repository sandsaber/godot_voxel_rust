//! Threading primitives ported from `util/thread/`.
//!
//! The C++ layer exposes locks as standalone synchronization objects:
//! `Mutex` is recursive (`std::recursive_mutex`), `BinaryMutex` is non-recursive
//! (`std::mutex`), and `RWLock` wraps `std::shared_timed_mutex`. These Rust
//! wrappers keep the same lock-object shape and return RAII guards.

#[cfg(test)]
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering as AtomicOrdering};
use std::sync::{
    Condvar, Mutex as StdMutex, MutexGuard as StdMutexGuard, RwLock as StdRwLock,
    RwLockReadGuard as StdRwLockReadGuard, RwLockWriteGuard as StdRwLockWriteGuard, TryLockError,
};
use std::thread::ThreadId;

#[derive(Debug, Default)]
struct RecursiveState {
    owner: Option<ThreadId>,
    depth: usize,
}

/// Recursive mutex. Ported from C++ `Mutex` (`std::recursive_mutex`).
#[derive(Debug, Default)]
pub struct Mutex {
    state: StdMutex<RecursiveState>,
    cvar: Condvar,
}

impl Mutex {
    pub fn new() -> Self {
        Self::default()
    }

    /// Lock recursively on the current thread, blocking until available.
    pub fn lock(&self) -> MutexGuard<'_> {
        let current = std::thread::current().id();
        let mut state = lock_unpoisoned(&self.state);
        loop {
            match state.owner {
                None => {
                    state.owner = Some(current);
                    state.depth = 1;
                    return MutexGuard::new(self);
                }
                Some(owner) if owner == current => {
                    state.depth += 1;
                    return MutexGuard::new(self);
                }
                Some(_) => {
                    state = wait_unpoisoned(&self.cvar, state);
                }
            }
        }
    }

    /// Try to lock recursively on the current thread.
    pub fn try_lock(&self) -> Option<MutexGuard<'_>> {
        let current = std::thread::current().id();
        let mut state = match self.state.try_lock() {
            Ok(state) => state,
            Err(TryLockError::Poisoned(e)) => e.into_inner(),
            Err(TryLockError::WouldBlock) => return None,
        };
        match state.owner {
            None => {
                state.owner = Some(current);
                state.depth = 1;
                Some(MutexGuard::new(self))
            }
            Some(owner) if owner == current => {
                state.depth += 1;
                Some(MutexGuard::new(self))
            }
            Some(_) => None,
        }
    }

    fn unlock(&self) {
        let current = std::thread::current().id();
        let mut state = lock_unpoisoned(&self.state);
        debug_assert_eq!(state.owner, Some(current));
        debug_assert!(state.depth > 0);
        state.depth -= 1;
        if state.depth == 0 {
            state.owner = None;
            self.cvar.notify_one();
        }
    }
}

/// RAII guard returned by [`Mutex::lock`] / [`Mutex::try_lock`].
#[derive(Debug)]
pub struct MutexGuard<'a> {
    lock: &'a Mutex,
    _not_send: std::marker::PhantomData<std::rc::Rc<()>>,
}

impl<'a> MutexGuard<'a> {
    fn new(lock: &'a Mutex) -> Self {
        Self {
            lock,
            _not_send: std::marker::PhantomData,
        }
    }
}

impl Drop for MutexGuard<'_> {
    fn drop(&mut self) {
        self.lock.unlock();
    }
}

/// Non-recursive mutex. Ported from C++ `BinaryMutex` (`std::mutex`).
#[derive(Debug, Default)]
pub struct BinaryMutex {
    inner: StdMutex<()>,
}

impl BinaryMutex {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn lock(&self) -> BinaryMutexGuard<'_> {
        BinaryMutexGuard {
            _guard: lock_unpoisoned(&self.inner),
        }
    }

    pub fn try_lock(&self) -> Option<BinaryMutexGuard<'_>> {
        match self.inner.try_lock() {
            Ok(guard) => Some(BinaryMutexGuard { _guard: guard }),
            Err(TryLockError::Poisoned(e)) => Some(BinaryMutexGuard {
                _guard: e.into_inner(),
            }),
            Err(TryLockError::WouldBlock) => None,
        }
    }
}

/// RAII guard returned by [`BinaryMutex`].
#[derive(Debug)]
pub struct BinaryMutexGuard<'a> {
    _guard: StdMutexGuard<'a, ()>,
}

/// Read/write lock. Ported from C++ `RWLock` (`std::shared_timed_mutex`).
#[derive(Debug, Default)]
pub struct RwLock {
    inner: StdRwLock<()>,
}

impl RwLock {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn read_lock(&self) -> RwLockReadGuard<'_> {
        RwLockReadGuard {
            _guard: read_unpoisoned(&self.inner),
        }
    }

    pub fn read_try_lock(&self) -> Option<RwLockReadGuard<'_>> {
        match self.inner.try_read() {
            Ok(guard) => Some(RwLockReadGuard { _guard: guard }),
            Err(TryLockError::Poisoned(e)) => Some(RwLockReadGuard {
                _guard: e.into_inner(),
            }),
            Err(TryLockError::WouldBlock) => None,
        }
    }

    pub fn write_lock(&self) -> RwLockWriteGuard<'_> {
        RwLockWriteGuard {
            _guard: write_unpoisoned(&self.inner),
        }
    }

    pub fn write_try_lock(&self) -> Option<RwLockWriteGuard<'_>> {
        match self.inner.try_write() {
            Ok(guard) => Some(RwLockWriteGuard { _guard: guard }),
            Err(TryLockError::Poisoned(e)) => Some(RwLockWriteGuard {
                _guard: e.into_inner(),
            }),
            Err(TryLockError::WouldBlock) => None,
        }
    }
}

/// RAII read guard returned by [`RwLock`].
#[derive(Debug)]
pub struct RwLockReadGuard<'a> {
    _guard: StdRwLockReadGuard<'a, ()>,
}

/// RAII write guard returned by [`RwLock`].
#[derive(Debug)]
pub struct RwLockWriteGuard<'a> {
    _guard: StdRwLockWriteGuard<'a, ()>,
}

fn lock_unpoisoned<T>(mutex: &StdMutex<T>) -> StdMutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(|e| e.into_inner())
}

fn wait_unpoisoned<'a, T>(cvar: &Condvar, guard: StdMutexGuard<'a, T>) -> StdMutexGuard<'a, T> {
    cvar.wait(guard).unwrap_or_else(|e| e.into_inner())
}

fn read_unpoisoned<T>(lock: &StdRwLock<T>) -> StdRwLockReadGuard<'_, T> {
    lock.read().unwrap_or_else(|e| e.into_inner())
}

fn write_unpoisoned<T>(lock: &StdRwLock<T>) -> StdRwLockWriteGuard<'_, T> {
    lock.write().unwrap_or_else(|e| e.into_inner())
}

/// Counting semaphore. Ported from `util/thread/semaphore.h` (header-only,
/// built on `std::mutex` + `std::condition_variable`).
///
/// Hand-rolled with `Mutex<usize>` + `Condvar` to keep the crate dependency-
/// free (the stdlib `Semaphore` is unstable; `parking_lot::Semaphore` would
/// add a runtime dep).
#[derive(Debug, Default)]
pub struct Semaphore {
    state: StdMutex<usize>,
    cvar: Condvar,
}

impl Semaphore {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_count(count: usize) -> Self {
        Self {
            state: StdMutex::new(count),
            cvar: Condvar::new(),
        }
    }

    /// Increment the counter and wake one waiter.
    pub fn post(&self) {
        let mut count = lock_unpoisoned(&self.state);
        *count = count.saturating_add(1);
        self.cvar.notify_one();
    }

    /// Block until the counter is non-zero, then decrement it.
    pub fn wait(&self) {
        let mut count = lock_unpoisoned(&self.state);
        while *count == 0 {
            count = wait_unpoisoned(&self.cvar, count);
        }
        *count -= 1;
    }

    /// Decrement the counter if non-zero; returns `true` on success.
    pub fn try_wait(&self) -> bool {
        let mut count = lock_unpoisoned(&self.state);
        if *count == 0 {
            return false;
        }
        *count -= 1;
        true
    }

    pub fn count(&self) -> usize {
        *lock_unpoisoned(&self.state)
    }
}

/// Mode of a [`SpatialLock3D`] area guard. Mirrors C++ `SpatialLock3D::Mode`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpatialLockMode {
    Read,
    Write,
}

/// Region-based read/write lock over 3D integer boxes.
///
/// Ported from `util/thread/spatial_lock_3d.{h,cpp}`. Multiple read locks may
/// overlap; a write lock excludes every overlapping read or write lock.
/// Disjoint boxes can proceed concurrently.
#[derive(Debug, Default)]
pub struct SpatialLock3D {
    state: StdMutex<SpatialLockRegistry>,
    cvar: Condvar,
    #[cfg(test)]
    fail_write_registry_reservation: AtomicBool,
    #[cfg(test)]
    fail_write_registry_slot_reservation_after_active: AtomicBool,
    #[cfg(test)]
    single_write_waiters: AtomicUsize,
}

#[derive(Debug, Default)]
struct SpatialLockRegistry {
    // One stable slot represents one logical owner. A write batch therefore
    // publishes and retires as one unit regardless of its number of bounds.
    slots: Vec<SpatialLockSlot>,
    // Conflict checks visit live owners only; vacant slab slots never make a
    // later acquisition proportional to the historical peak owner count.
    active_slots: Vec<usize>,
    // Vacant slots form an intrusive free list, so retirement cannot allocate.
    first_free_slot: Option<usize>,
    locked_bounds: usize,
    #[cfg(test)]
    indexed_slot_inspections: usize,
}

#[derive(Debug)]
enum SpatialLockSlot {
    Occupied {
        entry: SpatialLockEntry,
        active_index: usize,
    },
    Vacant {
        next: Option<usize>,
    },
}

#[derive(Debug)]
enum SpatialLockEntry {
    Single {
        bounds: crate::math::BoxBounds3i,
        mode: SpatialLockMode,
    },
    WriteBatch {
        bounds: Vec<crate::math::BoxBounds3i>,
    },
}

impl SpatialLockEntry {
    fn bounds_count(&self) -> usize {
        match self {
            Self::Single { .. } => 1,
            Self::WriteBatch { bounds } => bounds.len(),
        }
    }

    fn can_acquire(&self, bounds: crate::math::BoxBounds3i, mode: SpatialLockMode) -> bool {
        match self {
            Self::Single {
                bounds: held_bounds,
                mode: held_mode,
            } => spatial_regions_can_coexist(bounds, mode, *held_bounds, *held_mode),
            Self::WriteBatch {
                bounds: held_bounds,
            } => held_bounds.iter().all(|held_bounds| {
                spatial_regions_can_coexist(bounds, mode, *held_bounds, SpatialLockMode::Write)
            }),
        }
    }
}

impl SpatialLockRegistry {
    fn can_acquire(&self, bounds: crate::math::BoxBounds3i, mode: SpatialLockMode) -> bool {
        self.active_slots
            .iter()
            .map(|slot| match &self.slots[*slot] {
                SpatialLockSlot::Occupied { entry, .. } => entry,
                SpatialLockSlot::Vacant { .. } => {
                    unreachable!("active spatial registry slot must be occupied")
                }
            })
            .all(|entry| entry.can_acquire(bounds, mode))
    }

    fn write_batch_can_acquire(&self, bounds: &[crate::math::BoxBounds3i]) -> bool {
        bounds
            .iter()
            .all(|bounds| self.can_acquire(*bounds, SpatialLockMode::Write))
    }

    fn reserve_slot(&mut self) {
        self.active_slots.reserve(1);
        if self.first_free_slot.is_none() {
            self.slots.reserve(1);
        }
    }

    fn try_reserve_slot(&mut self, #[cfg(test)] fail_after_active_reservation: bool) -> bool {
        if self.active_slots.try_reserve(1).is_err() {
            return false;
        }
        #[cfg(test)]
        if fail_after_active_reservation {
            return false;
        }
        if self.first_free_slot.is_none() && self.slots.try_reserve(1).is_err() {
            return false;
        }
        true
    }

    fn insert_reserved(&mut self, entry: SpatialLockEntry) -> usize {
        let bounds_count = entry.bounds_count();
        let next_locked_bounds = self
            .locked_bounds
            .checked_add(bounds_count)
            .expect("live spatial lock count cannot exceed addressable memory");
        let active_index = self.active_slots.len();
        let slot = if let Some(slot) = self.first_free_slot {
            let SpatialLockSlot::Vacant { next } = &self.slots[slot] else {
                unreachable!("free spatial registry slot must be vacant")
            };
            let next = *next;
            self.slots[slot] = SpatialLockSlot::Occupied {
                entry,
                active_index,
            };
            self.first_free_slot = next;
            slot
        } else {
            debug_assert!(self.slots.len() < self.slots.capacity());
            let slot = self.slots.len();
            self.slots.push(SpatialLockSlot::Occupied {
                entry,
                active_index,
            });
            slot
        };
        debug_assert!(self.active_slots.len() < self.active_slots.capacity());
        self.active_slots.push(slot);
        self.locked_bounds = next_locked_bounds;
        slot
    }

    fn remove(&mut self, slot: usize) -> SpatialLockEntry {
        #[cfg(test)]
        {
            self.indexed_slot_inspections += 1;
        }
        let Some(target) = self.slots.get(slot) else {
            unreachable!("spatial guard slot must belong to this registry")
        };
        let SpatialLockSlot::Occupied {
            active_index,
            entry,
        } = target
        else {
            unreachable!("spatial guard must own a live registry slot")
        };
        let active_index = *active_index;
        let next_locked_bounds = self
            .locked_bounds
            .checked_sub(entry.bounds_count())
            .expect("spatial registry count must include every live owner");
        let moved_slot = self.active_slots.last().copied();
        let first_free_slot = self.first_free_slot;
        let old = std::mem::replace(
            &mut self.slots[slot],
            SpatialLockSlot::Vacant {
                next: first_free_slot,
            },
        );
        let SpatialLockSlot::Occupied { entry, .. } = old else {
            unreachable!("spatial guard must own a live registry slot")
        };
        self.active_slots.swap_remove(active_index);
        if active_index < self.active_slots.len() {
            let moved_slot = moved_slot.expect("swap removal must have a final active slot");
            #[cfg(test)]
            {
                self.indexed_slot_inspections += 1;
            }
            let SpatialLockSlot::Occupied {
                active_index: moved_active_index,
                ..
            } = &mut self.slots[moved_slot]
            else {
                unreachable!("moved active spatial registry slot must be occupied")
            };
            *moved_active_index = active_index;
        }
        self.first_free_slot = Some(slot);
        self.locked_bounds = next_locked_bounds;
        entry
    }
}

/// Canonical write regions prepared before entering a higher-level mutation
/// critical section.
///
/// Keeping the owned bounds in this value lets callers perform collection,
/// sorting, and deduplication before acquiring their mutation gate. Acquiring
/// the spatial lock then only publishes the already-prepared regions.
#[derive(Debug)]
pub(crate) struct PreparedSpatialWriteBatch {
    bounds: Vec<crate::math::BoxBounds3i>,
}

impl SpatialLock3D {
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns `false` if an overlapping box is currently held for writing.
    pub fn try_lock_read(&self, bounds: crate::math::BoxBounds3i) -> bool {
        let mut entries = lock_unpoisoned(&self.state);
        if entries.can_acquire(bounds, SpatialLockMode::Read) {
            entries.reserve_slot();
            entries.insert_reserved(SpatialLockEntry::Single {
                bounds,
                mode: SpatialLockMode::Read,
            });
            true
        } else {
            false
        }
    }

    /// Block until no overlapping write lock exists.
    pub fn lock_read(&self, bounds: crate::math::BoxBounds3i) {
        let mut entries = lock_unpoisoned(&self.state);
        while !entries.can_acquire(bounds, SpatialLockMode::Read) {
            entries = wait_unpoisoned(&self.cvar, entries);
        }
        entries.reserve_slot();
        entries.insert_reserved(SpatialLockEntry::Single {
            bounds,
            mode: SpatialLockMode::Read,
        });
    }

    pub fn unlock_read(&self, bounds: crate::math::BoxBounds3i) {
        self.unlock(bounds, SpatialLockMode::Read);
    }

    /// Returns `false` if any overlapping read or write lock exists.
    pub fn try_lock_write(&self, bounds: crate::math::BoxBounds3i) -> bool {
        let mut entries = lock_unpoisoned(&self.state);
        if entries.can_acquire(bounds, SpatialLockMode::Write) {
            entries.reserve_slot();
            entries.insert_reserved(SpatialLockEntry::Single {
                bounds,
                mode: SpatialLockMode::Write,
            });
            true
        } else {
            false
        }
    }

    /// Block until no overlapping lock exists.
    pub fn lock_write(&self, bounds: crate::math::BoxBounds3i) {
        let mut entries = lock_unpoisoned(&self.state);
        while !entries.can_acquire(bounds, SpatialLockMode::Write) {
            #[cfg(test)]
            self.single_write_waiters
                .fetch_add(1, AtomicOrdering::SeqCst);
            entries = wait_unpoisoned(&self.cvar, entries);
            #[cfg(test)]
            self.single_write_waiters
                .fetch_sub(1, AtomicOrdering::SeqCst);
        }
        entries.reserve_slot();
        entries.insert_reserved(SpatialLockEntry::Single {
            bounds,
            mode: SpatialLockMode::Write,
        });
    }

    pub fn unlock_write(&self, bounds: crate::math::BoxBounds3i) {
        self.unlock(bounds, SpatialLockMode::Write);
    }

    pub fn locked_boxes_count(&self) -> usize {
        lock_unpoisoned(&self.state).locked_bounds
    }

    fn unlock(&self, bounds: crate::math::BoxBounds3i, mode: SpatialLockMode) {
        let mut entries = lock_unpoisoned(&self.state);
        let Some(index) = entries.active_slots.iter().copied().find(|slot| {
            matches!(
                &entries.slots[*slot],
                SpatialLockSlot::Occupied {
                    entry: SpatialLockEntry::Single {
                        bounds: held_bounds,
                        mode: held_mode,
                    },
                    ..
                } if *held_bounds == bounds && *held_mode == mode
            )
        }) else {
            debug_assert_eq!(
                entries
                    .active_slots
                    .iter()
                    .filter(|slot| {
                        matches!(
                            &entries.slots[**slot],
                            SpatialLockSlot::Occupied {
                                entry: SpatialLockEntry::Single {
                                    bounds: held_bounds,
                                    mode: held_mode,
                                },
                                ..
                            } if *held_bounds == bounds && *held_mode == mode
                        )
                    })
                    .count(),
                1,
                "unlock called for a SpatialLock3D entry that is not held"
            );
            return;
        };
        entries.remove(index);
        self.cvar.notify_all();
    }

    /// Convenience: acquire a read lock for `bounds` and return an RAII guard.
    /// Mirrors the C++ `SpatialLock3D::Read` nested type.
    pub fn read(&self, bounds: crate::math::BoxBounds3i) -> SpatialLockReadGuard<'_> {
        self.lock_read(bounds);
        SpatialLockReadGuard { lock: self, bounds }
    }

    /// Convenience: acquire a write lock for `bounds` and return an RAII guard.
    /// Mirrors the C++ `SpatialLock3D::Write` nested type.
    pub fn write(&self, bounds: crate::math::BoxBounds3i) -> SpatialLockWriteGuard<'_> {
        self.lock_write(bounds);
        SpatialLockWriteGuard { lock: self, bounds }
    }

    /// Acquires one logical write batch atomically.
    ///
    /// Empty boxes are ignored and exact duplicates are removed. The remaining
    /// bounds are stored in canonical coordinate order. Conflicts are checked
    /// only against guards that existed before this batch, so overlapping
    /// regions owned by the same batch cannot deadlock each other.
    pub fn write_many(
        &self,
        bounds: impl IntoIterator<Item = crate::math::BoxBounds3i>,
    ) -> SpatialLockWriteManyGuard<'_> {
        self.write_prepared(Self::prepare_write_many(bounds))
    }

    /// Collects and canonicalizes one logical write batch without acquiring
    /// this spatial lock.
    pub(crate) fn prepare_write_many(
        bounds: impl IntoIterator<Item = crate::math::BoxBounds3i>,
    ) -> PreparedSpatialWriteBatch {
        PreparedSpatialWriteBatch {
            bounds: canonical_spatial_bounds(bounds),
        }
    }

    /// Fallible counterpart of [`Self::prepare_write_many`].
    ///
    /// Higher-level transactions use this to keep every recoverable allocation
    /// outside their mutation gate.
    #[allow(dead_code)] // Prepared shared-data transactions adopt this before terrain cutover.
    pub(crate) fn try_prepare_write_many(
        bounds: impl IntoIterator<Item = crate::math::BoxBounds3i>,
    ) -> Result<PreparedSpatialWriteBatch, std::collections::TryReserveError> {
        let bounds = bounds.into_iter();
        let (_, upper_bound) = bounds.size_hint();
        let mut canonical = Vec::new();
        if let Some(upper_bound) = upper_bound {
            canonical.try_reserve_exact(upper_bound)?;
        }
        for bounds in bounds {
            if bounds.is_empty() {
                continue;
            }
            if canonical.len() == canonical.capacity() {
                canonical.try_reserve(1)?;
            }
            canonical.push(bounds);
        }
        canonicalize_spatial_bounds(&mut canonical);
        Ok(PreparedSpatialWriteBatch { bounds: canonical })
    }

    /// Acquires a logical write batch that was canonicalized in advance.
    pub(crate) fn write_prepared(
        &self,
        prepared: PreparedSpatialWriteBatch,
    ) -> SpatialLockWriteManyGuard<'_> {
        let bounds = prepared.bounds;
        let mut entries = lock_unpoisoned(&self.state);
        while !entries.write_batch_can_acquire(&bounds) {
            entries = wait_unpoisoned(&self.cvar, entries);
        }
        // Publishing one logical batch must never grow the live registry.
        // Non-retryable callers retain their historical infallible API, while
        // retryable transactions use `write_prepared_fallible` below.
        entries.reserve_slot();
        let slot = entries.insert_reserved(SpatialLockEntry::WriteBatch { bounds });
        SpatialLockWriteManyGuard {
            lock: self,
            slot: Some(slot),
        }
    }

    /// Acquires a prepared batch while keeping registry allocation failure
    /// retryable.
    ///
    /// The live registry is reserved while its mutex is held, immediately
    /// before publication. On failure the exact owned batch is returned and
    /// no registry entry has been added.
    pub(crate) fn write_prepared_fallible(
        &self,
        prepared: PreparedSpatialWriteBatch,
    ) -> Result<SpatialLockWriteManyGuard<'_>, PreparedSpatialWriteBatch> {
        let bounds = prepared.bounds;
        let mut entries = lock_unpoisoned(&self.state);
        while !entries.write_batch_can_acquire(&bounds) {
            entries = wait_unpoisoned(&self.cvar, entries);
        }
        #[cfg(test)]
        if self
            .fail_write_registry_reservation
            .load(AtomicOrdering::SeqCst)
        {
            return Err(PreparedSpatialWriteBatch { bounds });
        }
        if !entries.try_reserve_slot(
            #[cfg(test)]
            self.fail_write_registry_slot_reservation_after_active
                .load(AtomicOrdering::SeqCst),
        ) {
            return Err(PreparedSpatialWriteBatch { bounds });
        }
        let slot = entries.insert_reserved(SpatialLockEntry::WriteBatch { bounds });
        Ok(SpatialLockWriteManyGuard {
            lock: self,
            slot: Some(slot),
        })
    }

    #[cfg(test)]
    pub(crate) fn set_test_write_registry_reservation_failure(&self, should_fail: bool) {
        self.fail_write_registry_reservation
            .store(should_fail, AtomicOrdering::SeqCst);
    }

    #[cfg(test)]
    fn set_test_write_registry_slot_reservation_failure_after_active(&self, should_fail: bool) {
        self.fail_write_registry_slot_reservation_after_active
            .store(should_fail, AtomicOrdering::SeqCst);
    }

    #[cfg(test)]
    fn reset_test_release_indexed_slot_inspections(&self) {
        lock_unpoisoned(&self.state).indexed_slot_inspections = 0;
    }

    #[cfg(test)]
    fn test_release_indexed_slot_inspections(&self) -> usize {
        lock_unpoisoned(&self.state).indexed_slot_inspections
    }

    #[cfg(test)]
    fn test_single_write_waiters(&self) -> usize {
        self.single_write_waiters.load(AtomicOrdering::SeqCst)
    }

    /// Attempts to acquire one logical write batch without blocking.
    pub fn try_write_many(
        &self,
        bounds: impl IntoIterator<Item = crate::math::BoxBounds3i>,
    ) -> Option<SpatialLockWriteManyGuard<'_>> {
        self.try_write_prepared(Self::prepare_write_many(bounds))
    }

    /// Attempts to acquire a logical write batch that was canonicalized in
    /// advance, without blocking.
    pub(crate) fn try_write_prepared(
        &self,
        prepared: PreparedSpatialWriteBatch,
    ) -> Option<SpatialLockWriteManyGuard<'_>> {
        let bounds = prepared.bounds;
        let mut entries = lock_unpoisoned(&self.state);
        if !entries.write_batch_can_acquire(&bounds) {
            return None;
        }
        entries.reserve_slot();
        let slot = entries.insert_reserved(SpatialLockEntry::WriteBatch { bounds });
        Some(SpatialLockWriteManyGuard {
            lock: self,
            slot: Some(slot),
        })
    }
}

/// RAII read guard for [`SpatialLock3D`]. Releases on drop.
#[derive(Debug)]
pub struct SpatialLockReadGuard<'a> {
    lock: &'a SpatialLock3D,
    bounds: crate::math::BoxBounds3i,
}

impl Drop for SpatialLockReadGuard<'_> {
    fn drop(&mut self) {
        self.lock.unlock_read(self.bounds);
    }
}

/// RAII write guard for [`SpatialLock3D`]. Releases on drop.
#[derive(Debug)]
pub struct SpatialLockWriteGuard<'a> {
    lock: &'a SpatialLock3D,
    bounds: crate::math::BoxBounds3i,
}

impl Drop for SpatialLockWriteGuard<'_> {
    fn drop(&mut self) {
        self.lock.unlock_write(self.bounds);
    }
}

/// RAII guard for one atomically-acquired [`SpatialLock3D`] write batch.
#[derive(Debug)]
pub struct SpatialLockWriteManyGuard<'a> {
    lock: &'a SpatialLock3D,
    slot: Option<usize>,
}

impl SpatialLockWriteManyGuard<'_> {
    /// Releases this guard and returns the exact canonical prepared batch.
    ///
    /// This is used only by retryable higher-level transactions. It performs
    /// no allocation and leaves the batch ready for another ordered attempt.
    #[allow(dead_code)] // Used by retryable prepared storage commits.
    pub(crate) fn release_prepared(mut self) -> PreparedSpatialWriteBatch {
        let slot = self
            .slot
            .take()
            .expect("write batch guard owns its registry slot");
        let bounds = release_spatial_write_many(self.lock, slot);
        PreparedSpatialWriteBatch { bounds }
    }
}

impl Drop for SpatialLockWriteManyGuard<'_> {
    fn drop(&mut self) {
        if let Some(slot) = self.slot.take() {
            drop(release_spatial_write_many(self.lock, slot));
        }
    }
}

fn release_spatial_write_many(lock: &SpatialLock3D, slot: usize) -> Vec<crate::math::BoxBounds3i> {
    let mut entries = lock_unpoisoned(&lock.state);
    let entry = entries.remove(slot);
    lock.cvar.notify_all();
    drop(entries);
    match entry {
        SpatialLockEntry::WriteBatch { bounds } => bounds,
        SpatialLockEntry::Single { .. } => {
            unreachable!("write batch guard must own a batch registry slot")
        }
    }
}

fn canonical_spatial_bounds(
    bounds: impl IntoIterator<Item = crate::math::BoxBounds3i>,
) -> Vec<crate::math::BoxBounds3i> {
    let mut bounds = bounds
        .into_iter()
        .filter(|bounds| !bounds.is_empty())
        .collect::<Vec<_>>();
    canonicalize_spatial_bounds(&mut bounds);
    bounds
}

fn canonicalize_spatial_bounds(bounds: &mut Vec<crate::math::BoxBounds3i>) {
    bounds.sort_unstable_by_key(|bounds| {
        (
            bounds.min_pos.x,
            bounds.min_pos.y,
            bounds.min_pos.z,
            bounds.max_pos.x,
            bounds.max_pos.y,
            bounds.max_pos.z,
        )
    });
    bounds.dedup();
}

fn spatial_regions_can_coexist(
    requested_bounds: crate::math::BoxBounds3i,
    requested_mode: SpatialLockMode,
    held_bounds: crate::math::BoxBounds3i,
    held_mode: SpatialLockMode,
) -> bool {
    !held_bounds.intersects(&requested_bounds)
        || matches!(
            (requested_mode, held_mode),
            (SpatialLockMode::Read, SpatialLockMode::Read)
        )
}

#[cfg(test)]
mod tests {
    use super::{
        BinaryMutex, Mutex, RwLock, SpatialLockEntry, SpatialLockRegistry, SpatialLockSlot,
    };
    use std::sync::mpsc;
    use std::sync::Arc;
    use std::time::Duration;

    fn assert_spatial_registry_matches_model(
        registry: &SpatialLockRegistry,
        model: &[Option<usize>],
    ) {
        assert_eq!(registry.slots.len(), model.len());

        let mut active_seen = vec![false; registry.slots.len()];
        let mut locked_bounds = 0usize;
        for (active_index, slot) in registry.active_slots.iter().copied().enumerate() {
            assert!(slot < registry.slots.len(), "active slot must be in range");
            assert!(!active_seen[slot], "active slots must be unique");
            active_seen[slot] = true;
            let SpatialLockSlot::Occupied {
                entry,
                active_index: backlink,
            } = &registry.slots[slot]
            else {
                panic!("every active slot must be occupied");
            };
            assert_eq!(*backlink, active_index, "active backlink must be exact");
            assert_eq!(model[slot], Some(entry.bounds_count()));
            locked_bounds += entry.bounds_count();
        }

        let mut free_seen = vec![false; registry.slots.len()];
        let mut cursor = registry.first_free_slot;
        while let Some(slot) = cursor {
            assert!(slot < registry.slots.len(), "free slot must be in range");
            assert!(!free_seen[slot], "free-list must not contain a cycle");
            free_seen[slot] = true;
            let SpatialLockSlot::Vacant { next } = &registry.slots[slot] else {
                panic!("every free-list node must be vacant");
            };
            cursor = *next;
        }

        for (slot, expected) in model.iter().enumerate() {
            match (&registry.slots[slot], expected) {
                (SpatialLockSlot::Occupied { .. }, Some(_)) => {
                    assert!(active_seen[slot]);
                    assert!(!free_seen[slot]);
                }
                (SpatialLockSlot::Vacant { .. }, None) => {
                    assert!(!active_seen[slot]);
                    assert!(free_seen[slot], "free-list must cover every vacant slot");
                }
                _ => panic!("registry slot occupancy must match the model"),
            }
        }
        assert_eq!(
            registry.active_slots.len(),
            model.iter().filter(|entry| entry.is_some()).count()
        );
        assert_eq!(registry.locked_bounds, locked_bounds);
        assert_eq!(
            registry.locked_bounds,
            model.iter().flatten().copied().sum::<usize>()
        );
    }

    fn model_entry(id: i32, bounds_count: usize) -> SpatialLockEntry {
        use crate::math::{BoxBounds3i, Vector3i};

        if bounds_count == 1 && id % 2 == 0 {
            let min = Vector3i::new(id * 100, 0, 0);
            return SpatialLockEntry::Single {
                bounds: BoxBounds3i::new(min, min + Vector3i::splat(1)),
                mode: super::SpatialLockMode::Read,
            };
        }
        SpatialLockEntry::WriteBatch {
            bounds: (0..bounds_count)
                .map(|offset| {
                    let min = Vector3i::new(id * 100 + offset as i32 * 2, 0, 0);
                    BoxBounds3i::new(min, min + Vector3i::splat(1))
                })
                .collect(),
        }
    }

    #[test]
    fn mutex_allows_recursive_locking_on_same_thread() {
        let lock = Mutex::new();
        let _outer = lock.lock();
        assert!(lock.try_lock().is_some());
    }

    #[test]
    fn binary_mutex_try_lock_fails_while_held() {
        let lock = BinaryMutex::new();
        let _guard = lock.lock();
        assert!(lock.try_lock().is_none());
    }

    #[test]
    fn rw_lock_allows_multiple_readers_but_excludes_writer() {
        let lock = Arc::new(RwLock::new());
        let read_a = lock.read_lock();
        let read_b = lock.read_lock();
        assert!(lock.write_try_lock().is_none());
        drop(read_a);
        assert!(lock.write_try_lock().is_none());
        drop(read_b);
        assert!(lock.write_try_lock().is_some());
    }

    #[test]
    fn rw_lock_writer_excludes_readers_across_threads() {
        let lock = Arc::new(RwLock::new());
        let writer = lock.write_lock();
        let (tx, rx) = mpsc::channel();
        let worker_lock = lock.clone();

        let handle = std::thread::spawn(move || {
            tx.send(worker_lock.read_try_lock().is_none()).unwrap();
        });

        assert!(rx.recv_timeout(Duration::from_secs(1)).unwrap());
        drop(writer);
        handle.join().unwrap();
        assert!(lock.read_try_lock().is_some());
    }

    #[test]
    fn semaphore_try_wait_returns_false_at_zero_and_true_after_post() {
        use super::Semaphore;
        let sem = Semaphore::new();
        assert_eq!(sem.count(), 0);
        assert!(!sem.try_wait());
        sem.post();
        sem.post();
        assert_eq!(sem.count(), 2);
        assert!(sem.try_wait());
        assert_eq!(sem.count(), 1);
    }

    #[test]
    fn semaphore_wait_blocks_until_another_thread_posts() {
        use super::Semaphore;
        let sem = Arc::new(Semaphore::new());
        let worker_sem = sem.clone();
        let handle = std::thread::spawn(move || {
            // Worker waits (will block until the main thread posts).
            worker_sem.wait();
        });
        // Give the worker a moment to enter wait(), then post.
        std::thread::sleep(Duration::from_millis(20));
        sem.post();
        handle.join().expect("worker should unblock after post");
    }

    #[test]
    fn spatial_lock_3d_respects_overlap_and_mode() {
        use super::SpatialLock3D;
        use crate::math::{BoxBounds3i, Vector3i};
        let lock = SpatialLock3D::new();
        let area = BoxBounds3i::new(Vector3i::zero(), Vector3i::new(4, 4, 4));
        let overlap = BoxBounds3i::new(Vector3i::new(2, 2, 2), Vector3i::new(6, 6, 6));
        let disjoint = BoxBounds3i::new(Vector3i::new(8, 8, 8), Vector3i::new(10, 10, 10));

        let read = lock.read(area);
        assert!(lock.try_lock_read(overlap), "overlapping reads may coexist");
        assert_eq!(lock.locked_boxes_count(), 2);
        assert!(
            !lock.try_lock_write(overlap),
            "overlapping write must wait for readers"
        );
        assert!(
            lock.try_lock_write(disjoint),
            "disjoint write can run alongside reads"
        );
        assert_eq!(lock.locked_boxes_count(), 3);
        lock.unlock_write(disjoint);
        lock.unlock_read(overlap);
        drop(read);

        assert!(lock.try_lock_write(overlap));
        assert_eq!(lock.locked_boxes_count(), 1);
        lock.unlock_write(overlap);
        assert_eq!(lock.locked_boxes_count(), 0);
    }

    #[test]
    fn spatial_lock_3d_blocking_write_waits_for_overlapping_read() {
        use super::SpatialLock3D;
        use crate::math::{BoxBounds3i, Vector3i};

        let lock = Arc::new(SpatialLock3D::new());
        let bounds = BoxBounds3i::new(Vector3i::zero(), Vector3i::new(4, 4, 4));
        let read = lock.read(bounds);
        let worker_lock = lock.clone();
        let (attempt_tx, attempt_rx) = mpsc::channel();
        let (acquired_tx, acquired_rx) = mpsc::channel();

        let handle = std::thread::spawn(move || {
            attempt_tx.send(()).unwrap();
            let _write = worker_lock.write(bounds);
            acquired_tx.send(()).unwrap();
        });

        attempt_rx.recv_timeout(Duration::from_secs(1)).unwrap();
        assert!(
            acquired_rx.recv_timeout(Duration::from_millis(50)).is_err(),
            "writer acquired overlapping region before read guard was dropped"
        );
        drop(read);
        acquired_rx.recv_timeout(Duration::from_secs(1)).unwrap();
        handle.join().unwrap();
    }

    #[test]
    fn spatial_write_many_canonicalizes_one_atomic_owner_batch() {
        use super::SpatialLock3D;
        use crate::math::{BoxBounds3i, Vector3i};

        let lock = SpatialLock3D::new();
        let first = BoxBounds3i::new(Vector3i::zero(), Vector3i::new(4, 4, 4));
        let overlapping = BoxBounds3i::new(Vector3i::new(2, 2, 2), Vector3i::new(6, 6, 6));
        let empty = BoxBounds3i::new(Vector3i::splat(7), Vector3i::splat(7));

        let batch = lock.write_many([overlapping, first, overlapping, empty]);
        assert_eq!(lock.locked_boxes_count(), 2);
        assert!(!lock.try_lock_read(first));
        assert!(lock.try_write_many([first]).is_none());

        drop(batch);
        assert_eq!(lock.locked_boxes_count(), 0);
        assert!(lock.try_write_many([first]).is_some());
    }

    #[test]
    fn prepared_spatial_write_batch_canonicalizes_before_acquire() {
        use super::SpatialLock3D;
        use crate::math::{BoxBounds3i, Vector3i};

        let lock = SpatialLock3D::new();
        let first = BoxBounds3i::new(Vector3i::zero(), Vector3i::new(4, 4, 4));
        let overlapping = BoxBounds3i::new(Vector3i::new(2, 2, 2), Vector3i::new(6, 6, 6));
        let empty = BoxBounds3i::new(Vector3i::splat(7), Vector3i::splat(7));

        let prepared = SpatialLock3D::prepare_write_many([overlapping, first, overlapping, empty]);
        assert_eq!(prepared.bounds, vec![first, overlapping]);

        let batch = lock.write_prepared(prepared);
        assert_eq!(lock.locked_boxes_count(), 2);
        assert!(!lock.try_lock_read(first));

        drop(batch);
        assert_eq!(lock.locked_boxes_count(), 0);
    }

    #[test]
    fn spatial_write_many_waits_for_the_complete_batch_before_publication() {
        use super::SpatialLock3D;
        use crate::math::{BoxBounds3i, Vector3i};

        let lock = Arc::new(SpatialLock3D::new());
        let blocked = BoxBounds3i::new(Vector3i::zero(), Vector3i::new(4, 4, 4));
        let free = BoxBounds3i::new(Vector3i::splat(10), Vector3i::splat(14));
        let read = lock.read(blocked);
        let worker_lock = lock.clone();
        let (acquired_tx, acquired_rx) = mpsc::channel();
        let worker = std::thread::spawn(move || {
            let batch = worker_lock.write_many([free, blocked]);
            acquired_tx.send(()).unwrap();
            drop(batch);
        });

        assert!(acquired_rx.recv_timeout(Duration::from_millis(50)).is_err());
        assert!(lock.try_lock_read(free));
        lock.unlock_read(free);
        drop(read);
        acquired_rx.recv_timeout(Duration::from_secs(1)).unwrap();
        worker.join().unwrap();
        assert_eq!(lock.locked_boxes_count(), 0);
    }

    #[test]
    fn prepared_spatial_write_batch_release_is_constant_registry_work() {
        use super::SpatialLock3D;
        use crate::math::{BoxBounds3i, Vector3i};

        const BATCH_SIZE: i32 = 512;
        let lock = SpatialLock3D::new();
        let bounds = (0..BATCH_SIZE)
            .map(|x| {
                let min = Vector3i::new(x * 2, 0, 0);
                BoxBounds3i::new(min, min + Vector3i::splat(1))
            })
            .collect::<Vec<_>>();
        let prepared = SpatialLock3D::prepare_write_many(bounds.iter().copied());
        let allocation = prepared.bounds.as_ptr();
        let guard = lock.write_prepared(prepared);
        let unrelated = (0..128)
            .map(|x| {
                let min = Vector3i::new(10_000 + x * 2, 0, 0);
                BoxBounds3i::new(min, min + Vector3i::splat(1))
            })
            .collect::<Vec<_>>();
        for bounds in &unrelated {
            assert!(lock.try_lock_read(*bounds));
        }

        lock.reset_test_release_indexed_slot_inspections();
        let prepared = guard.release_prepared();

        assert_eq!(prepared.bounds.as_ptr(), allocation);
        assert_eq!(prepared.bounds, bounds);
        assert_eq!(
            lock.test_release_indexed_slot_inspections(),
            2,
            "non-last batch release must inspect only its slot and the moved slot"
        );
        assert_eq!(lock.locked_boxes_count(), unrelated.len());
        let moved_owner = *unrelated
            .last()
            .expect("large regression fixture has unrelated owners");
        assert!(
            !lock.try_lock_write(moved_owner),
            "non-last removal must preserve the owner moved in the dense index"
        );
        lock.unlock_read(moved_owner);
        assert_eq!(lock.locked_boxes_count(), unrelated.len() - 1);
        assert!(lock.try_lock_read(moved_owner));
        assert_eq!(lock.locked_boxes_count(), unrelated.len());

        let guard = lock.write_prepared(prepared);
        assert_eq!(
            lock.locked_boxes_count(),
            BATCH_SIZE as usize + unrelated.len()
        );
        drop(guard);
        for bounds in unrelated {
            lock.unlock_read(bounds);
        }
        assert_eq!(lock.locked_boxes_count(), 0);
    }

    #[test]
    fn prepared_spatial_write_batch_reservation_failure_returns_exact_owner() {
        use super::SpatialLock3D;
        use crate::math::{BoxBounds3i, Vector3i};

        let lock = SpatialLock3D::new();
        let bounds = [
            BoxBounds3i::new(Vector3i::zero(), Vector3i::splat(1)),
            BoxBounds3i::new(Vector3i::splat(4), Vector3i::splat(5)),
        ];
        let prepared = SpatialLock3D::prepare_write_many(bounds);
        let allocation = prepared.bounds.as_ptr();
        lock.set_test_write_registry_reservation_failure(true);

        let prepared = match lock.write_prepared_fallible(prepared) {
            Ok(_) => panic!("injected registry reservation failure must be retryable"),
            Err(prepared) => prepared,
        };

        assert_eq!(prepared.bounds.as_ptr(), allocation);
        assert_eq!(prepared.bounds, bounds);
        assert_eq!(lock.locked_boxes_count(), 0);

        lock.set_test_write_registry_reservation_failure(false);
        let guard = lock
            .write_prepared_fallible(prepared)
            .expect("same prepared owner must be accepted on retry");
        let prepared = guard.release_prepared();
        assert_eq!(prepared.bounds.as_ptr(), allocation);
        assert_eq!(prepared.bounds, bounds);
        assert_eq!(lock.locked_boxes_count(), 0);
    }

    #[test]
    fn spatial_registry_permutations_preserve_dense_and_free_list_invariants() {
        let mut registry = SpatialLockRegistry::default();
        let mut model = Vec::<Option<usize>>::new();

        for id in 0..12 {
            let bounds_count = id as usize % 4 + 1;
            registry.reserve_slot();
            let slot = registry.insert_reserved(model_entry(id, bounds_count));
            assert_eq!(slot, model.len());
            model.push(Some(bounds_count));
            assert_spatial_registry_matches_model(&registry, &model);
        }

        let mut freed_slots = Vec::new();
        for _ in 0..3 {
            for selector in 0..3 {
                let active_index = match selector {
                    0 => 0,
                    1 => registry.active_slots.len() / 2,
                    _ => registry.active_slots.len() - 1,
                };
                let slot = registry.active_slots[active_index];
                let expected_count = model[slot]
                    .take()
                    .expect("selected model owner must be live");
                let removed = registry.remove(slot);
                assert_eq!(removed.bounds_count(), expected_count);
                freed_slots.push(slot);
                assert_spatial_registry_matches_model(&registry, &model);
            }
        }

        for (reuse_index, expected_slot) in freed_slots.iter().rev().copied().enumerate() {
            let bounds_count = reuse_index % 3 + 1;
            registry.reserve_slot();
            let slot =
                registry.insert_reserved(model_entry(100 + reuse_index as i32, bounds_count));
            assert_eq!(
                slot, expected_slot,
                "free-list reuse must be LIFO and exact"
            );
            model[slot] = Some(bounds_count);
            assert_spatial_registry_matches_model(&registry, &model);
        }

        let mut selector = 0usize;
        while !registry.active_slots.is_empty() {
            let active_index = match selector % 3 {
                0 => 0,
                1 => registry.active_slots.len() / 2,
                _ => registry.active_slots.len() - 1,
            };
            let slot = registry.active_slots[active_index];
            let expected_count = model[slot]
                .take()
                .expect("selected model owner must be live");
            assert_eq!(registry.remove(slot).bounds_count(), expected_count);
            assert_spatial_registry_matches_model(&registry, &model);
            selector += 1;
        }
    }

    #[test]
    fn prepared_spatial_write_batch_partial_reservation_failure_restores_content() {
        use super::SpatialLock3D;
        use crate::math::{BoxBounds3i, Vector3i};

        let lock = SpatialLock3D::new();
        let bounds = [BoxBounds3i::new(Vector3i::zero(), Vector3i::splat(2))];
        let prepared = SpatialLock3D::prepare_write_many(bounds);
        let allocation = prepared.bounds.as_ptr();
        let (initial_active_capacity, initial_slot_capacity) = {
            let registry = lock.state.lock().unwrap();
            (registry.active_slots.capacity(), registry.slots.capacity())
        };
        lock.set_test_write_registry_slot_reservation_failure_after_active(true);

        let prepared = match lock.write_prepared_fallible(prepared) {
            Ok(_) => panic!("second reservation failpoint must return the prepared owner"),
            Err(prepared) => prepared,
        };

        let registry = lock.state.lock().unwrap();
        assert!(registry.active_slots.capacity() > initial_active_capacity);
        assert_eq!(registry.slots.capacity(), initial_slot_capacity);
        assert!(registry.active_slots.is_empty());
        assert!(registry.slots.is_empty());
        assert_eq!(registry.first_free_slot, None);
        assert_eq!(registry.locked_bounds, 0);
        drop(registry);
        assert_eq!(prepared.bounds.as_ptr(), allocation);
        assert_eq!(prepared.bounds, bounds);

        lock.set_test_write_registry_slot_reservation_failure_after_active(false);
        let guard = lock
            .write_prepared_fallible(prepared)
            .expect("unchanged owner must remain retryable");
        assert_eq!(lock.locked_boxes_count(), 1);
        let prepared = guard.release_prepared();
        assert_eq!(prepared.bounds.as_ptr(), allocation);
        assert_eq!(prepared.bounds, bounds);
        assert_eq!(lock.locked_boxes_count(), 0);
    }

    #[test]
    fn spatial_write_batch_drop_wakes_blocked_single_waiter() {
        use super::SpatialLock3D;
        use crate::math::{BoxBounds3i, Vector3i};

        let lock = Arc::new(SpatialLock3D::new());
        let bounds = BoxBounds3i::new(Vector3i::zero(), Vector3i::splat(4));
        let batch = lock.write_many([bounds]);
        let worker_lock = Arc::clone(&lock);
        let (attempted_tx, attempted_rx) = mpsc::channel();
        let (acquired_tx, acquired_rx) = mpsc::channel();
        let worker = std::thread::spawn(move || {
            attempted_tx.send(()).unwrap();
            let single = worker_lock.write(bounds);
            acquired_tx.send(()).unwrap();
            drop(single);
        });

        attempted_rx.recv_timeout(Duration::from_secs(1)).unwrap();
        let wait_deadline = std::time::Instant::now() + Duration::from_secs(1);
        while lock.test_single_write_waiters() != 1 {
            assert!(
                std::time::Instant::now() < wait_deadline,
                "single writer must enter the condvar wait before batch drop"
            );
            std::thread::yield_now();
        }
        assert!(acquired_rx.try_recv().is_err());
        drop(batch);
        acquired_rx.recv_timeout(Duration::from_secs(1)).unwrap();
        worker.join().unwrap();
        assert_eq!(lock.test_single_write_waiters(), 0);
        assert_eq!(lock.locked_boxes_count(), 0);
    }
}
