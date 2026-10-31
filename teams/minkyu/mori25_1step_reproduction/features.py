"""Encode Ama afterstates without exposing current/NEXT/NEXT2 puyo pairs.

Ama stores colors only in the bottom 13 rows. Its 14th row is a six-bit
occupancy mask, not six color values. Therefore the input is 5 x 14 x 6:
channels 0-3 are R/Y/G/B; channel 4 is garbage (rows 0-12) or row-14
occupancy (row 13). This is NOT identical to the 2025 paper's 14x6x4.
"""

COLOR_CHANNEL = {"R": 0, "Y": 1, "G": 2, "B": 3, "#": 4}
CHANNELS = 5
HEIGHT = 14
WIDTH = 6


def encode_afterstate(observation):
    """Return nested Python lists [5][14][6], bottom row at index 0.

    Only board and row14 are used. The observation's `visible` pairs and
    `turn` are deliberately ignored to match NEXT-free value learning.
    """
    board = observation["board"]
    row14 = int(observation["row14"])
    if len(board) != 13 or not 0 <= row14 < 64:
        raise ValueError("Invalid Ama observation shape or row14 mask")

    planes = [[[0.0 for _ in range(WIDTH)] for _ in range(HEIGHT)]
              for _ in range(CHANNELS)]
    for y, row in enumerate(board):
        if len(row) != WIDTH:
            raise ValueError(f"Invalid row length: row {y}")
        for x, color in enumerate(row):
            if color == ".":
                continue
            if color not in COLOR_CHANNEL:
                raise ValueError(f"Unknown Puyo cell: {color!r}")
            planes[COLOR_CHANNEL[color]][y][x] = 1.0

    for x in range(WIDTH):
        if row14 & (1 << x):
            planes[4][13][x] = 1.0
    return planes
