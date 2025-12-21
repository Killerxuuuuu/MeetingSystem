def merge_segments(raw_segments, merge_threshold=1.5):
    """
    Manual implementation of the Segment Merging Algorithm.
    
    The sliding window approach creates many small, overlapping segments.
    This function merges adjacent segments that belong to the same speaker.
    
    Args:
        raw_segments: List of raw segments from the sliding window.
        merge_threshold: Max gap (in seconds) allowed to merge two segments.
        
    Returns:
        List of merged segments with continuous timelines.
    """
    if not raw_segments:
        return []
    
    merged = []
    # Start with the first segment
    current_seg = raw_segments[0].copy()
    
    for next_seg in raw_segments[1:]:
        # Logic for merging:
        # 1. The speaker ID must be the same.
        # 2. The start of the next segment is within the range of (current_end + threshold).
        # Note: Since windows overlap, next_start is usually < current_end.
        if (next_seg['speaker'] == current_seg['speaker'] and 
            next_seg['start'] <= current_seg['end'] + merge_threshold):
            
            # Extend the current segment
            current_seg['end'] = max(current_seg['end'], next_seg['end'])
        else:
            # Found a new speaker or a long silence gap.
            # Save the current segment and start a new one.
            merged.append(current_seg)
            current_seg = next_seg.copy()
            
    # Don't forget to append the last segment
    merged.append(current_seg) 
    return merged