use vstd::prelude::*;

verus! {
    // Direct translation of the Dafny method as a spec function
    spec fn update_map<K, V>(m1: Map<K, V>, m2: Map<K, V>) -> Map<K, V> {
        Map::new(
            |k: K| m1.dom().contains(k) || m2.dom().contains(k),
            |k: K| if m2.dom().contains(k) { m2.index(k) } else { m1.index(k) }
        )
    }
    
    // Properties that match the original Dafny ensures clauses
    proof fn update_map_ensures<K, V>(m1: Map<K, V>, m2: Map<K, V>)
        ensures
            // forall k :: k in m2 ==> k in r
            forall |k: K| m2.dom().contains(k) ==> update_map(m1, m2).dom().contains(k),
            // forall k :: k in m1 ==> k in r
            forall |k: K| m1.dom().contains(k) ==> update_map(m1, m2).dom().contains(k),
            // forall k :: k in m2 ==> r[k] == m2[k]
            forall |k: K| m2.dom().contains(k) ==> update_map(m1, m2).index(k) == m2.index(k),
            // forall k :: !(k in m2) && k in m1 ==> r[k] == m1[k]
            forall |k: K| !m2.dom().contains(k) && m1.dom().contains(k) ==> update_map(m1, m2).index(k) == m1.index(k),
            // forall k :: !(k in m2) && !(k in m1) ==> !(k in r)
            forall |k: K| !m2.dom().contains(k) && !m1.dom().contains(k) ==> !update_map(m1, m2).dom().contains(k),
    {
        // These properties follow directly from the definition of update_map
    }
}