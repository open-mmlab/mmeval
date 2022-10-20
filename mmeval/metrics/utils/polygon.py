# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
from typing import TYPE_CHECKING, Optional, Sequence, Tuple, Union

from mmeval.utils import try_import

if TYPE_CHECKING:
    from shapely.geometry import MultiPolygon, Polygon
else:
    geometry = try_import('shapely.geometry')
    if geometry is not None:
        Polygon = geometry.Polygon
        MultiPolygon = geometry.MultiPolygon

ArrayLike = Union[np.ndarray, Sequence[Union[int, float]]]


def poly2shapely(polygon: ArrayLike) -> 'Polygon':
    """Convert a polygon to shapely.geometry.Polygon.

    Args:
        polygon (ArrayLike): A set of points of 2k shape.

    Returns:
        polygon (Polygon): A polygon object.
    """
    polygon = np.array(polygon, dtype=np.float32)
    assert polygon.size % 2 == 0 and polygon.size >= 6

    polygon = polygon.reshape([-1, 2])
    return Polygon(polygon)


def polys2shapely(polygons: Sequence[ArrayLike]) -> Sequence['Polygon']:
    """Convert a nested list of boundaries to a list of Polygons.

    Args:
        polygons (list): The point coordinates of the instance boundary.

    Returns:
        list: Converted shapely.Polygon.
    """
    return [poly2shapely(polygon) for polygon in polygons]


def poly_make_valid(poly: 'Polygon') -> 'Polygon':
    """Convert a potentially invalid polygon to a valid one by eliminating
    self-crossing or self-touching parts.

    Args:
        poly (Polygon): A polygon needed to be converted.

    Returns:
        Polygon: A valid polygon.
    """
    assert isinstance(poly, Polygon)
    return poly if poly.is_valid else poly.buffer(0)


def poly_intersection(
        poly_a: 'Polygon',
        poly_b: 'Polygon',
        invalid_ret: Optional[Union[float, int]] = None,
        return_poly: bool = False) -> Tuple[float, Optional['Polygon']]:
    """Calculate the intersection area between two polygons.

    Args:
        poly_a (Polygon): Polygon a.
        poly_b (Polygon): Polygon b.
        invalid_ret (float or int, optional): The return value when the
            invalid polygon exists. If it is not specified, the function
            allows the computation to proceed with invalid polygons by
            cleaning the their self-touching or self-crossing parts.
            Defaults to None.
        return_poly (bool): Whether to return the polygon of the intersection
            Defaults to False.

    Returns:
        float or tuple(float, Polygon): Returns the intersection area or
        a tuple ``(area, Optional[poly_obj])``, where the `area` is the
        intersection area between two polygons and `poly_obj` is The Polygon
        object of the intersection area. Set as `None` if the input is invalid.
        Set as `None` if the input is invalid. `poly_obj` will be returned
        only if `return_poly` is `True`.
    """
    assert isinstance(poly_a, Polygon)
    assert isinstance(poly_b, Polygon)
    assert invalid_ret is None or isinstance(invalid_ret, (float, int))

    if invalid_ret is None:
        poly_a = poly_make_valid(poly_a)
        poly_b = poly_make_valid(poly_b)

    poly_obj = None
    area = invalid_ret
    if poly_a.is_valid and poly_b.is_valid:
        poly_obj = poly_a.intersection(poly_b)
        area = poly_obj.area
    return (area, poly_obj) if return_poly else area  # type: ignore


def poly_union(
    poly_a: 'Polygon',
    poly_b: 'Polygon',
    invalid_ret: Optional[Union[float, int]] = None,
    return_poly: bool = False
) -> Tuple[float, Optional[Union['Polygon', 'MultiPolygon']]]:
    """Calculate the union area between two polygons.

    Args:
        poly_a (Polygon): Polygon a.
        poly_b (Polygon): Polygon b.
        invalid_ret (float or int, optional): The return value when the
            invalid polygon exists. If it is not specified, the function
            allows the computation to proceed with invalid polygons by
            cleaning the their self-touching or self-crossing parts.
            Defaults to False.
        return_poly (bool): Whether to return the polygon of the union.
            Defaults to False.

    Returns:
        tuple: Returns a tuple ``(area, Optional[poly_obj])``, where
        the `area` is the union between two polygons and `poly_obj` is the
        Polygon or MultiPolygon object of the union of the inputs. The type
        of object depends on whether they intersect or not. Set as `None`
        if the input is invalid. `poly_obj` will be returned only if
        `return_poly` is `True`.
    """
    assert isinstance(poly_a, Polygon)
    assert isinstance(poly_b, Polygon)
    assert invalid_ret is None or isinstance(invalid_ret, (float, int))

    if invalid_ret is None:
        poly_a = poly_make_valid(poly_a)
        poly_b = poly_make_valid(poly_b)

    poly_obj = None
    area = invalid_ret
    if poly_a.is_valid and poly_b.is_valid:
        poly_obj = poly_a.union(poly_b)
        area = poly_obj.area
    return (area, poly_obj) if return_poly else area  # type: ignore


def poly_iou(poly_a: 'Polygon',
             poly_b: 'Polygon',
             zero_division: float = 0.) -> float:
    """Calculate the IOU between two polygons.

    Args:
        poly_a (Polygon): Polygon a.
        poly_b (Polygon): Polygon b.
        zero_division (float): The return value when invalid polygon exists.

    Returns:
        float: The IoU between two polygons.
    """
    assert isinstance(poly_a, Polygon)
    assert isinstance(poly_b, Polygon)
    area_inters = poly_intersection(poly_a, poly_b)
    area_union = poly_union(poly_a, poly_b)
    return area_inters / area_union if area_union != 0 else zero_division  # type:ignore # noqa
