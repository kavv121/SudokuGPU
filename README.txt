After building, there are a few modes:
./sudokusolver
Reads one 9x9 puzzle as a string of 81 characters (using 0 or '.' for blank) from stdin
./sudokusolver file.txt [sidesize]
Reads strings of (sidesize**2)x(sidesize**2) puzzles from the given file and solves the entire batch.  Default sidesize is 3 (i.e. 9x9 puzzles).
3,4, or 5 should work

TEST CASES
In addition to the test cases in the various test directories, there is also the 17-clue normalized, unique solution database of almost 50k 9x9 puzzles collected by Gordon Royle that can be found at http://staffhome.ecm.uwa.edu.au/~00013890/sudoku17 (not included in this repository).  The file can be passed to this solver as-is.
