# =====================================
# One cell's regions, arcs and boundary segments, out of the cache's flat lists
#
# Host-side conveniences over the cache's own storage, returned as views: on a device cache, copy
# what you need to the host first (`Array(cache.info)`, ...).

"""
    region(cache::PolylineClippingCutCellCache, ci, r) -> CutCellData

Fluid region `r` of cell `ci`, whatever the cell: the cell's own record when it has one region, the
`r`-th of its region records when it is split. Regions are numbered in walk order from the cell's
lower-left corner.
"""
function region(cache::PolylineClippingCutCellCache, ci::CartesianIndex{2}, r::Integer)
    inf = cache.info[ci]
    1 <= r <= inf.nregion || throw(BoundsError(1:inf.nregion, r))
    return inf.nregion == 1 ? cache.cells[ci] : cache.regions[inf.rslot + r - 1]
end

"""
    regions(cache::PolylineClippingCutCellCache, ci) -> AbstractVector{CutCellData}

A split cell's region records, in walk order; empty for any other cell (whose one region, if it has
one, is `cache.cells[ci]` -- or use [`region`](@ref)).
"""
function regions(cache::PolylineClippingCutCellCache, ci::CartesianIndex{2})
    inf = cache.info[ci]
    return inf.nregion >= 2 ? view(cache.regions, inf.rslot:(inf.rslot + inf.nregion - 1)) :
                              view(cache.regions, 1:0)
end

# The cut slot a cell had in the last update, 0 if none.
_pl_slot_of(cache::PolylineClippingCutCellCache, ci::CartesianIndex{2}) =
    _pl_slot_of(cache.info.slot, cache.work.cross.cut_list, ci)

# The same from the `slot` column itself, so a host copy can stand in for a device cache's
# (`interface_mesh`).
function _pl_slot_of(slot::AbstractMatrix, cut_list::AbstractVector, ci::CartesianIndex{2})
    s = Int(slot[ci])
    return s > 0 && s <= length(cut_list) && cut_list[s] == ci ? s : 0
end

"""
    arcs(cache::PolylineClippingCutCellCache, ci) -> AbstractVector{PolylineArc}

The fluid intervals of cut cell `ci`'s four edges, grouped by direction, with the region owning each
on either side; empty for an uncut cell.
"""
function arcs(cache::PolylineClippingCutCellCache, ci::CartesianIndex{2})
    s = _pl_slot_of(cache, ci)
    C = cache.work.cross
    return s == 0 ? view(cache.arcs, 1:0) : view(cache.arcs, C.arc_start[s]:(C.arc_start[s + 1] - 1))
end

"""
    boundary_segments(cache::PolylineClippingCutCellCache, ci) -> AbstractVector{PolylineBoundarySeg}

The pieces of the elements inside cut cell `ci`, each with the region whose wall it is (`0` for a
sliver's, dropped); empty for an uncut cell.
"""
function boundary_segments(cache::PolylineClippingCutCellCache, ci::CartesianIndex{2})
    s = _pl_slot_of(cache, ci)
    C = cache.work.cross
    return s == 0 ? view(cache.bsegs, 1:0) : view(cache.bsegs, C.bseg_start[s]:(C.bseg_start[s + 1] - 1))
end

# The same, by linear cell index -- what `findfirst` over one of the cache's arrays hands back.
for f in (:region, :regions, :arcs, :boundary_segments)
    @eval $f(cache::PolylineClippingCutCellCache, i::Integer, args...) =
        $f(cache, CartesianIndices(size(cache.cells))[i], args...)
end
