"""
Mapping between geometry CSV names and actual image filenames
"""

def map_geometry_name(name: str, column: str = None) -> str:
    """
    Map a name from the geometry CSV to the actual image filename.
    
    Args:
        name: The name from the CSV file
        column: Optional column name ('Sample', 'Correct', 'Incorrect') for context
        
    Returns:
        The mapped filename or the original name if no mapping found
    """
    if not name or str(name) == 'nan':
        return name
        
    name = str(name).strip()
    
    # If it already looks like a complete filename, return as-is
    if name.endswith('.jpg') or name.endswith('.png'):
        return name
    
    # Determine if we need big or small based on column
    size = 'big' if column == 'Sample' else 'small'
    
    # Handle different naming patterns
    
    # Pattern 1: quad1, quad2, quad3, quad4, etc. -> a_big_1, a_big_2, etc.
    if name.startswith('quad'):
        number = name.replace('quad', '')
        if number.isdigit():
            return f"a_{size}_{number}.jpg"
    
    # Pattern 2: Single letters A, B, C, D, E -> ad_big_A, ad_big_B, etc.
    if len(name) == 1 and name in 'ABCDE':
        return f"ad_{size}_{name}.jpg"
    
    # Pattern 3: p1, p2, p3, etc. -> p_big_1, p_big_2, etc.
    if name.startswith('p') and len(name) == 2 and name[1].isdigit():
        return f"p_{size}_{name[1]}.jpg"
    
    # Pattern 4: s1, s2, s3, etc. -> a_big_1, a_big_2, etc. (s = square = a)
    if name.startswith('s') and len(name) == 2 and name[1].isdigit():
        return f"a_{size}_{name[1]}.jpg"
    
    # Pattern 5: Just numbers 1, 2, 3, etc. -> a_small_1, a_small_2, etc.
    if name.isdigit():
        return f"a_small_{name}.jpg"
    
    # Pattern 6: triangle, pentagon, decagon, heptagon, hexagon, octagon
    shape_mapping = {
        'triangle': f"t_{size}_triangle.jpg",
        'pentagon': f"t_{size}_pentagon.jpg",
        'decagon': f"t_{size}_decagon.jpg",
        'heptagon': f"t_{size}_irreg_heptagon.jpg",
        'hexagon': f"t_{size}_irreg_hexagon.jpg",
        'octagon': f"t_{size}_irreg_octagon.jpg"
    }
    if name in shape_mapping:
        return shape_mapping[name]
    
    # Pattern 7: d_A, d_B, etc. -> ad_big_A, ad_big_B, etc.
    if name.startswith('d_') and len(name) == 3:
        letter = name[2]
        return f"ad_{size}_{letter}.jpg"
    
    # Pattern 8: Already formatted names like a_big_1, p_small_2, etc.
    if '_' in name:
        # Check if it needs .jpg extension
        if not name.endswith('.jpg'):
            return f"{name}.jpg"
        return name
    
    # Default: return with .jpg extension
    return f"{name}.jpg"


# Original static mapping for reference/fallback
GEOMETRY_NAME_MAPPING = {
    # Sample column mappings (big versions)
    'quad1': 'a_big_1.jpg',
    'quad2': 'a_big_2.jpg',
    'quad3': 'a_big_3.jpg',
    'quad4': 'a_big_4.jpg',
    'quad5': 'a_big_5.jpg',
    'quad6': 'a_big_6.jpg',
    'quad7': 'a_big_7.jpg',
    'A': 'ad_big_A.jpg',
    'B': 'ad_big_B.jpg',
    'C': 'ad_big_C.jpg',
    'D': 'ad_big_D.jpg',
    'E': 'ad_big_E.jpg',
    'p1': 'p_big_1.jpg',
    'p2': 'p_big_2.jpg',
    'p3': 'p_big_3.jpg',
    'p4': 'p_big_4.jpg',
    'p5': 'p_big_5.jpg',
    'p6': 'p_big_6.jpg',
    'p7': 'p_big_7.jpg',
    's1': 'a_big_1.jpg',
    's2': 'a_big_2.jpg',
    's3': 'a_big_3.jpg',
    's4': 'a_big_4.jpg',
    's5': 'a_big_5.jpg',
    's6': 'a_big_6.jpg',
    's7': 'a_big_7.jpg',
    'triangle': 't_big_triangle.jpg',
    'pentagon': 't_big_pentagon.jpg',
    'decagon': 't_big_decagon.jpg',
    'heptagon': 't_big_irreg_heptagon.jpg',
    'hexagon': 't_big_irreg_hexagon.jpg',
    'octagon': 't_big_irreg_octagon.jpg',
    'd_A': 'ad_big_A.jpg',
    'd_B': 'ad_big_B.jpg',
    'd_C': 'ad_big_C.jpg',
    'd_D': 'ad_big_D.jpg',
    'd_E': 'ad_big_E.jpg',
    
    # Small versions for L and R columns (Correct/Incorrect)
    # Numbers alone typically refer to small 'a' shapes
    '1': 'a_small_1.jpg',
    '2': 'a_small_2.jpg',
    '3': 'a_small_3.jpg',
    '4': 'a_small_4.jpg',
    '5': 'a_small_5.jpg',
    '6': 'a_small_6.jpg',
    '7': 'a_small_7.jpg',
}