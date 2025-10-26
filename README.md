# Tetable

Tool for debugging and generating lookup tables for slicing tetrahedra

## Lehmer Ranks

When we split the tetrahedra, the neighboring tetrehedra must agree on how the shared polygons are triangulated. Otherwise the overall grid won't be valid. The basic idea for achieving this is for all tetrahedra in the grid to agree on the same rule for triangulating the quadrilaterals. The rule is: Always place the diagonal at the original tet-vertex (ignore edge points) that has the largest global vertex index. Because the global indices are shared, that means the neighboring tetrahedra will triangulate the shared face in the same way.

Now, that still doesn't fully explain how the lookup tables should work. We can't have a lookup table entry for every permutation of every 4 indices out of the total number of vertices in the grid. We rely on the fact that the we don't actually need to know the exact global indices to achive consistent triangulation. We only need to know their order, i.e. which vertex has the larger global index in a polygon. This is where Lehmer ranks come in. Say you iterate over the vertices of the tetrahedron according to their local order, and list their global indices. And say, the indices are strictly increasing. The Lehmer rank of such a tet is 0. If the global indices are strictly decreasing, the Lehmer rank is 23. There are 24 total permutations (hence 0 to 23), and each permutation has a unique Lehmer rank, that encodes the ordering of the vertex global indices. The Lehmer rank is enough to decide which vertex to prefer when triangulating a polygon. Since there are only 24 possible Lehmer ranks, we can use them in a lookup table.
