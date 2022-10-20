import numpy as np
import pytest

from mmeval.metrics.utils import (poly2shapely, poly_intersection, poly_iou,
                                  poly_make_valid, poly_union, polys2shapely)
from mmeval.utils import try_import

geometry = try_import('shapely.geometry')
if geometry is not None:
    Polygon = geometry.Polygon
    MultiPolygon = geometry.MultiPolygon

torch = try_import('torch')


@pytest.mark.skipif(geometry is None, reason='shapely is not available!')
def test_poly2shapely():
    polygon = Polygon([[0, 0], [1, 0], [1, 1], [0, 1]])
    # test np.array
    poly = np.array([0, 0, 1, 0, 1, 1, 0, 1])
    assert poly2shapely(poly) == polygon
    # test list
    poly = [0, 0, 1, 0, 1, 1, 0, 1]
    assert poly2shapely(poly) == polygon
    # test tensor
    poly = torch.Tensor([0, 0, 1, 0, 1, 1, 0, 1])
    assert poly2shapely(poly) == polygon
    # test invalid
    poly = [0, 0, 1]
    with pytest.raises(AssertionError):
        poly2shapely(poly)
    poly = [0, 0, 1, 0, 1, 1, 0, 1, 1]
    with pytest.raises(AssertionError):
        poly2shapely(poly)


@pytest.mark.skipif(geometry is None, reason='shapely is not available!')
def test_polys2shapely():
    polygons = [
        Polygon([[0, 0], [1, 0], [1, 1], [0, 1]]),
        Polygon([[1, 0], [1, 1], [0, 1], [0, 0]])
    ]
    # test np.array
    polys = np.array([[0, 0, 1, 0, 1, 1, 0, 1], [1, 0, 1, 1, 0, 1, 0, 0]])
    assert polys2shapely(polys) == polygons
    # test list
    polys = [[0, 0, 1, 0, 1, 1, 0, 1], [1, 0, 1, 1, 0, 1, 0, 0]]
    assert polys2shapely(polys) == polygons
    # test tensor
    polys = torch.Tensor([[0, 0, 1, 0, 1, 1, 0, 1], [1, 0, 1, 1, 0, 1, 0, 0]])
    assert polys2shapely(polys) == polygons
    # test invalid
    polys = [0, 0, 1]
    with pytest.raises(AssertionError):
        polys2shapely(polys)
    polys = [0, 0, 1, 0, 1, 1, 0, 1, 1]
    with pytest.raises(AssertionError):
        polys2shapely(polys)


@pytest.mark.skipif(geometry is None, reason='shapely is not available!')
def test_poly_make_valid():
    poly = Polygon([[0, 0], [1, 1], [1, 0], [0, 1]])
    assert not poly.is_valid
    poly = poly_make_valid(poly)
    assert poly.is_valid
    # invalid input
    with pytest.raises(AssertionError):
        poly_make_valid([0, 0, 1, 1, 1, 0, 0, 1])


@pytest.mark.skipif(geometry is None, reason='shapely is not available!')
def test_poly_intersection():

    # test unsupported type
    with pytest.raises(AssertionError):
        poly_intersection(0, 1)

    # test non-overlapping polygons
    points = [0, 0, 0, 1, 1, 1, 1, 0]
    points1 = [10, 20, 30, 40, 50, 60, 70, 80]
    points2 = [0, 0, 0, 0, 0, 0, 0, 0]  # Invalid polygon
    points3 = [0, 0, 0, 1, 1, 0, 1, 1]  # Self-intersected polygon
    points4 = [0.5, 0, 1.5, 0, 1.5, 1, 0.5, 1]
    poly = poly2shapely(points)
    poly1 = poly2shapely(points1)
    poly2 = poly2shapely(points2)
    poly3 = poly2shapely(points3)
    poly4 = poly2shapely(points4)

    area_inters = poly_intersection(poly, poly1)
    assert area_inters == 0

    # test overlapping polygons
    area_inters = poly_intersection(poly, poly)
    assert area_inters == 1
    area_inters = poly_intersection(poly, poly4)
    assert area_inters == 0.5

    # test invalid polygons
    poly_intersection(poly2, poly2) == 0
    poly_intersection(poly3, poly3, invalid_ret=1) == 1
    poly_intersection(poly3, poly3, invalid_ret=None) == 0.25

    # test poly return
    _, poly = poly_intersection(poly, poly4, return_poly=True)
    assert isinstance(poly, Polygon) is True
    _, poly = poly_intersection(
        poly3, poly3, invalid_ret=None, return_poly=True)
    assert isinstance(poly, Polygon) is True
    _, poly = poly_intersection(poly2, poly3, invalid_ret=1, return_poly=True)
    assert poly is None


@pytest.mark.skipif(geometry is None, reason='shapely is not available!')
def test_poly_union():

    # test unsupported type
    with pytest.raises(AssertionError):
        poly_union(0, 1)

    # test non-overlapping polygons

    points = [0, 0, 0, 1, 1, 1, 1, 0]
    points1 = [2, 2, 2, 3, 3, 3, 3, 2]
    points2 = [0, 0, 0, 0, 0, 0, 0, 0]  # Invalid polygon
    points3 = [0, 0, 0, 1, 1, 0, 1, 1]  # Self-intersected polygon
    points4 = [0.5, 0.5, 1, 0, 1, 1, 0.5, 0.5]
    poly = poly2shapely(points)
    poly1 = poly2shapely(points1)
    poly2 = poly2shapely(points2)
    poly3 = poly2shapely(points3)
    poly4 = poly2shapely(points4)

    assert poly_union(poly, poly1) == 2

    # test overlapping polygons
    assert poly_union(poly, poly) == 1

    # test invalid polygons
    assert poly_union(poly2, poly2) == 0
    assert poly_union(poly3, poly3, invalid_ret=1) == 1

    # The return value depends on the implementation of the package
    assert poly_union(poly3, poly3, invalid_ret=None) == 0.25
    assert poly_union(poly2, poly3) == 0.25
    assert poly_union(poly3, poly4) == 0.5

    # test poly return
    _, poly = poly_union(poly, poly1, return_poly=True)
    assert isinstance(poly, MultiPolygon) is True
    _, poly = poly_union(poly3, poly3, return_poly=True)
    assert isinstance(poly, Polygon) is True
    _, poly = poly_union(poly2, poly3, invalid_ret=0, return_poly=True)
    assert poly is None


@pytest.mark.skipif(geometry is None, reason='shapely is not available!')
def test_poly_iou():
    # test unsupported type
    with pytest.raises(AssertionError):
        poly_iou([1], [2])

    points = [0, 0, 0, 1, 1, 1, 1, 0]
    points1 = [10, 20, 30, 40, 50, 60, 70, 80]
    points2 = [0, 0, 0, 0, 0, 0, 0, 0]  # Invalid polygon
    points3 = [0, 0, 0, 1, 1, 0, 1, 1]  # Self-intersected polygon

    poly = poly2shapely(points)
    poly1 = poly2shapely(points1)
    poly2 = poly2shapely(points2)
    poly3 = poly2shapely(points3)

    assert poly_iou(poly, poly1) == 0

    # test overlapping polygons
    assert poly_iou(poly, poly) == 1

    # test invalid polygons
    assert poly_iou(poly2, poly2) == 0
    assert poly_iou(poly3, poly3, zero_division=1) == 1
    assert poly_iou(poly2, poly3) == 0
