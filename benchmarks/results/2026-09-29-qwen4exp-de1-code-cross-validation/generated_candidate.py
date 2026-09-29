def reconcile(events):
    """
    Reconcile room booking events.
    
    Args:
        events: List of event dictionaries with keys:
            - booking_id (str)
            - revision (int)
            - room (str)
            - start (int)
            - end (int)
            - cancelled (bool)
    
    Returns:
        Dictionary with 'accepted' and 'rejected' lists of booking IDs.
    """
    # Step 1: Select winning event for each booking_id
    # Largest revision wins; ties broken by later appearance in input
    winners = {}  # booking_id -> (revision, index, event)
    
    for idx, event in enumerate(events):
        bid = event['booking_id']
        rev = event['revision']
        
        if bid not in winners:
            winners[bid] = (rev, idx, event)
        else:
            old_rev, old_idx, old_event = winners[bid]
            # Larger revision wins; if tie, later index wins
            if rev > old_rev or (rev == old_rev and idx > old_idx):
                winners[bid] = (rev, idx, event)
    
    # Step 2: Filter out cancelled bookings
    surviving = []
    for bid, (rev, idx, event) in winners.items():
        if not event['cancelled']:
            surviving.append(event)
    
    # Step 3: Sort surviving bookings by (start, end, booking_id)
    surviving.sort(key=lambda e: (e['start'], e['end'], e['booking_id']))
    
    # Step 4: Process bookings, accepting unless overlap in same room
    accepted = []
    rejected = []
    
    # Track accepted bookings per room for overlap checking
    # For each room, keep list of accepted intervals (start, end)
    room_intervals = {}  # room -> list of (start, end)
    
    for event in surviving:
        room = event['room']
        start = event['start']
        end = event['end']
        bid = event['booking_id']
        
        if room not in room_intervals:
            room_intervals[room] = []
        
        # Check for overlap with any previously accepted booking in the same room
        # Overlap occurs if intervals intersect: not (end <= other_start or start >= other_end)
        # i.e., start < other_end and other_start < end
        has_overlap = False
        for other_start, other_end in room_intervals[room]:
            if start < other_end and other_start < end:
                has_overlap = True
                break
        
        if has_overlap:
            rejected.append(bid)
        else:
            accepted.append(bid)
            room_intervals[room].append((start, end))
    
    return {'accepted': accepted, 'rejected': rejected}


# Tests
def test_out_of_order_revisions():
    """Test that out-of-order revisions are handled correctly."""
    events = [
        {'booking_id': 'A', 'revision': 2, 'room': 'R1', 'start': 0, 'end': 10, 'cancelled': False},
        {'booking_id': 'A', 'revision': 1, 'room': 'R1', 'start': 0, 'end': 5, 'cancelled': False},
    ]
    result = reconcile(events)
    # Revision 2 wins, so A with [0,10) is accepted
    assert result['accepted'] == ['A']
    assert result['rejected'] == []


def test_equal_revision_last_arrival():
    """Test that when revisions tie, the later arrival wins."""
    events = [
        {'booking_id': 'A', 'revision': 1, 'room': 'R1', 'start': 0, 'end': 5, 'cancelled': False},
        {'booking_id': 'A', 'revision': 1, 'room': 'R1', 'start': 10, 'end': 15, 'cancelled': False},
    ]
    result = reconcile(events)
    # Second event (later arrival) wins, so A with [10,15) is accepted
    assert result['accepted'] == ['A']
    assert result['rejected'] == []


def test_cancellation_of_earlier_active():
    """Test that a cancelled winning event deletes the booking."""
    events = [
        {'booking_id': 'A', 'revision': 1, 'room': 'R1', 'start': 0, 'end': 10, 'cancelled': False},
        {'booking_id': 'A', 'revision': 2, 'room': 'R1', 'start': 0, 'end': 10, 'cancelled': True},
    ]
    result = reconcile(events)
    # Revision 2 wins and is cancelled, so A is deleted
    assert result['accepted'] == []
    assert result['rejected'] == []


def test_adjacency():
    """Test that adjacent bookings (end == start) do not overlap."""
    events = [
        {'booking_id': 'A', 'revision': 1, 'room': 'R1', 'start': 0, 'end': 5, 'cancelled': False},
        {'booking_id': 'B', 'revision': 1, 'room': 'R1', 'start': 5, 'end': 10, 'cancelled': False},
    ]
    result = reconcile(events)
    # Both should be accepted since they are adjacent
    assert result['accepted'] == ['A', 'B']
    assert result['rejected'] == []


def test_room_independence():
    """Test that different rooms are independent."""
    events = [
        {'booking_id': 'A', 'revision': 1, 'room': 'R1', 'start': 0, 'end': 10, 'cancelled': False},
        {'booking_id': 'B', 'revision': 1, 'room': 'R2', 'start': 0, 'end': 10, 'cancelled': False},
    ]
    result = reconcile(events)
    # Both should be accepted since they are in different rooms
    assert result['accepted'] == ['A', 'B']
    assert result['rejected'] == []


def test_rejected_does_not_block():
    """Test that a rejected booking does not block later bookings."""
    # A=[0,5), B=[3,10), C=[5,7) in same room
    # A is accepted, B overlaps with A so rejected, C does not overlap with A so accepted
    events = [
        {'booking_id': 'A', 'revision': 1, 'room': 'R1', 'start': 0, 'end': 5, 'cancelled': False},
        {'booking_id': 'B', 'revision': 1, 'room': 'R1', 'start': 3, 'end': 10, 'cancelled': False},
        {'booking_id': 'C', 'revision': 1, 'room': 'R1', 'start': 5, 'end': 7, 'cancelled': False},
    ]
    result = reconcile(events)
    # Processing order by (start, end, booking_id): A(0,5), B(3,10), C(5,7)
    # A accepted, B rejected (overlaps A), C accepted (no overlap with A)
    assert result['accepted'] == ['A', 'C']
    assert result['rejected'] == ['B']


def test_empty_input():
    """Test that empty input returns empty lists."""
    result = reconcile([])
    assert result['accepted'] == []
    assert result['rejected'] == []


def test_no_mutation():
    """Test that the input is not mutated."""
    import copy
    events = [
        {'booking_id': 'A', 'revision': 1, 'room': 'R1', 'start': 0, 'end': 10, 'cancelled': False},
        {'booking_id': 'B', 'revision': 1, 'room': 'R1', 'start': 5, 'end': 15, 'cancelled': False},
    ]
    original = copy.deepcopy(events)
    reconcile(events)
    assert events == original


if __name__ == '__main__':
    test_out_of_order_revisions()
    test_equal_revision_last_arrival()
    test_cancellation_of_earlier_active()
    test_adjacency()
    test_room_independence()
    test_rejected_does_not_block()
    test_empty_input()
    test_no_mutation()
    print("All tests passed!")
