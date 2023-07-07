use std::num::NonZeroU32;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct Generation(NonZeroU32);

impl Generation {
    fn new() -> Self {
        Self(unsafe { NonZeroU32::new_unchecked(1) })
    }

    fn next(self) -> Self {
        Self(self.0.checked_add(1).unwrap_or(unsafe { NonZeroU32::new_unchecked(1) }))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct NonZeroSlot(NonZeroU32);

impl NonZeroSlot {
    #[track_caller]
    fn new(slot: u32) -> Self {
        assert!(slot < u32::MAX, "slot must be less then u32::MAX");
        Self(unsafe { NonZeroU32::new_unchecked(slot + 1) })
    }

    fn get_slot(self) -> u32 {
        self.0.get() - 1
    }

    fn get_index(self) -> usize {
        self.0.get() as usize - 1
    }
}

#[derive(Debug, Clone)]
struct Entry<T> {
    generation: Generation,
    value: EntryValue<T>,
}

#[derive(Debug, Clone)]
enum EntryValue<T> {
    Occupied { val: T },
    Free { next_free: Option<NonZeroSlot> }
}

impl<T> EntryValue<T> {
    fn is_occupied(&self) -> bool {
        match self {
            EntryValue::Occupied { .. } => true,
            EntryValue::Free { .. } => false,
        }
    }

    fn take(&mut self, next_free: Option<NonZeroSlot>) -> Option<T> {
        if self.is_occupied() {
            let mut swap_value = Self::Free { next_free };
            std::mem::swap(&mut swap_value, self);
            
            let EntryValue::Occupied { val } = swap_value else {
                unreachable!()
            };
            
            Some(val)
        } else {
            None
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Index {
    generation: Generation,
    slot: NonZeroSlot,
}

impl Index {
    #[inline(always)]
    pub fn slot(self) -> u32 {
        self.slot.get_slot()
    }

    #[inline(always)]
    pub fn slot_index(self) -> usize {
        self.slot.get_index()
    }

    #[inline(always)]
    pub fn generation(self) -> u32 {
        self.generation.0.get()
    }
}

#[derive(Debug, Clone)]
pub struct Arena<T> {
    entries: Vec<Entry<T>>,
    first_free: Option<NonZeroSlot>,
    len: u32,
}

impl<T> Default for Arena<T> {
    fn default() -> Self {
        Self {
            entries: Vec::new(),
            first_free: None,
            len: 0,
        }
    }
}

impl<T> Arena<T> {
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
            first_free: None,
            len: 0,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty() && self.len == 0
    }

    pub fn with_capacity(capacity: u32) -> Self {
        Self {
            entries: Vec::with_capacity(capacity as usize),
            first_free: None,
            len: 0,
        }
    }

    pub fn len(&self) -> usize {
        self.len as usize
    }

    pub fn insert(&mut self, val: T) -> Index {
        assert!(self.len < u32::MAX, "cannot insert more then u32::MAX elements into arena");
        self.len += 1;
        if let Some(slot) = self.first_free.take() {
            let slot_index = slot.get_index();
            let EntryValue::Free { next_free } = self.entries[slot_index].value else {
                unreachable!()
            };
            self.first_free = next_free;
            let generation = self.entries[slot_index].generation.next();
            self.entries[slot_index] = Entry {
                generation,
                value: EntryValue::Occupied { val },
            };

            Index { generation, slot }
        } else {
            let generation = Generation::new();
            let slot = NonZeroSlot::new(self.entries.len() as u32);
            self.entries.push(Entry { generation, value: EntryValue::Occupied { val } });
            Index { generation, slot }
        }
    }

    pub fn get(&self, index: Index) -> Option<&T> {
        if let Some(Entry { generation, value: EntryValue::Occupied { val } }) = self.entries.get(index.slot.get_index()) {
            if *generation == index.generation {
                return Some(val);
            }
        }
        None
    }

    pub fn get_mut(&mut self, index: Index) -> Option<&mut T> {
        if let Some(Entry { generation, value: EntryValue::Occupied { val } }) = self.entries.get_mut(index.slot.get_index()) {
            if *generation == index.generation {
                return Some(val);
            }
        }
        None
    }

    pub fn remove(&mut self, index: Index) -> Option<T> {
        if let Some(Entry { generation, value }) = self.entries.get_mut(index.slot.get_index()) {
            if *generation == index.generation {
                let next_free = self.first_free;
                let val = value.take(next_free);
                if val.is_some() {
                    self.len -= 1;
                    self.first_free = Some(index.slot);
                }
                return val;
            }
        }
        None
    }

    pub fn clear(&mut self) {
        self.entries.clear();
        self.first_free = None;
        self.len = 0;
    }

    pub fn has_index(&self, index: Index) -> bool {
        if let Some(entry) = self.entries.get(index.slot.get_index()) {
            entry.value.is_occupied() && entry.generation == index.generation
        } else {
            false
        }
    }

    pub fn iter(&self) -> Iter<T> {
        Iter { entries: self.entries.iter(), slot: 0 }
    }

    pub fn iter_mut(&mut self) -> IterMut<T> {
        IterMut { entries: self.entries.iter_mut(), slot: 0 }
    }
}

pub struct Iter<'a, T> {
    entries: std::slice::Iter<'a, Entry<T>>,
    slot: u32
}

impl<'a, T> Iterator for Iter<'a, T> {
    type Item = (Index, &'a T);

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let entry = self.entries.next()?;
            let slot = self.slot;
            self.slot += 1;

            if let EntryValue::Occupied { ref val } = entry.value {
                return Some((Index { generation: entry.generation, slot: NonZeroSlot::new(slot) }, val));
            }
        }
    }
}

pub struct IterMut<'a, T> {
    entries: std::slice::IterMut<'a, Entry<T>>,
    slot: u32
}

impl<'a, T> Iterator for IterMut<'a, T> {
    type Item = (Index, &'a mut T);

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let entry = self.entries.next()?;
            let slot = self.slot;
            self.slot += 1;

            if let EntryValue::Occupied { ref mut val } = entry.value {
                return Some((Index { generation: entry.generation, slot: NonZeroSlot::new(slot) }, val));
            }
        }
    }
}

impl<T> std::ops::Index<Index> for Arena<T> {
    type Output = T;

    fn index(&self, index: Index) -> &Self::Output {
        self.get(index).unwrap()
    }
}

impl<T> std::ops::IndexMut<Index> for Arena<T> {
    fn index_mut(&mut self, index: Index) -> &mut Self::Output {
        self.get_mut(index).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn insert_get_len() {
        let mut arena = Arena::new();
        assert_eq!(arena.len, 0);

        let i0 = arena.insert(0);
        assert_eq!(arena.len, 1);
        let i1 = arena.insert(1);
        assert_eq!(arena.len, 2);
        let i2 = arena.insert(2);
        assert_eq!(arena.len, 3);

        assert_eq!(arena.get(i1), Some(&1));
        assert_eq!(arena.get(i0), Some(&0));
        assert_eq!(arena.get(i2), Some(&2));
    }

    #[test]
    fn insert_remove_len() {
        let mut arena = Arena::new();
        assert_eq!(arena.len, 0);

        let i0 = arena.insert(0);
        assert_eq!(arena.len(), 1);
        let i1 = arena.insert(1);
        assert_eq!(arena.len(), 2);
        let i2 = arena.insert(2);
        assert_eq!(arena.len(), 3);

        assert_eq!(arena.remove(i1), Some(1));
        assert_eq!(arena.len(), 2);
        assert_eq!(arena.remove(i1), None);
        assert_eq!(arena.len(), 2);

        assert_eq!(arena.remove(i0), Some(0));
        assert_eq!(arena.len(), 1);
        assert_eq!(arena.remove(i0), None);
        assert_eq!(arena.len(), 1);

        assert_eq!(arena.remove(i2), Some(2));
        assert_eq!(arena.len(), 0);
        assert_eq!(arena.remove(i2), None);
        assert_eq!(arena.len(), 0);
    }

    #[test]
    fn insert_remove_insert() {
        let mut arena = Arena::new();
        assert_eq!(arena.len, 0);

        let i0 = arena.insert(0);
        let i1 = arena.insert(1);
        let i2 = arena.insert(2);

        assert_eq!(arena.remove(i1), Some(1));
        assert_eq!(arena.remove(i1), None);

        let i3 = arena.insert(3);
        let v3 = arena.remove(i3);
        assert_eq!(v3, Some(3));

        assert_eq!(arena.remove(i0), Some(0));
        assert_eq!(arena.remove(i0), None);

        let i3 = arena.insert(3);
        assert_eq!(arena.remove(i3), Some(3));

        assert_eq!(arena.remove(i2), Some(2));
        assert_eq!(arena.remove(i2), None);

        let i3 = arena.insert(3);
        assert_eq!(arena.remove(i3), Some(3));
    }

    #[test]
    fn has_index() {
        let mut arena = Arena::new();
        
        let i0 = arena.insert(0);
        let i1 = arena.insert(1);
        let i2 = arena.insert(2);
        let i3 = arena.insert(3);
        let i4 = arena.insert(4);

        arena.remove(i1);
        arena.remove(i3);

        assert_eq!(arena.has_index(i0), true);
        assert_eq!(arena.has_index(i1), false);
        assert_eq!(arena.has_index(i2), true);
        assert_eq!(arena.has_index(i3), false);
        assert_eq!(arena.has_index(i4), true);
    }

    #[test]
    fn iter() {
        let mut arena = Arena::new();
        
        let i0 = arena.insert(0);
        let i1 = arena.insert(1);
        let i2 = arena.insert(2);
        let i3 = arena.insert(3);
        let i4 = arena.insert(4);

        arena.remove(i1);
        arena.remove(i3);

        let mut iter = arena.iter();

        assert_eq!(iter.next(), Some((i0, &0)));
        assert_eq!(iter.next(), Some((i2, &2)));
        assert_eq!(iter.next(), Some((i4, &4)));
    }
}