#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright (c) 2023 Raihan Kibria
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import random

WALL_NORTH = 0
WALL_SOUTH = 1
WALL_WEST = 2
WALL_EAST = 3

def labyrinth_to_string(labyrinth):
    diagram = ""
    diagram += " size: %s\n" % str(labyrinth["size"])
    diagram += "start: %s\n" % str(labyrinth["start"])
    diagram += " ends: %s\n" % str(labyrinth["ends"])

    cells = labyrinth["cells"]
    numRows = len(cells)
    for row in range(numRows):
        rowCells = cells[row]
        numCols = len(rowCells)

        if row == 0:
            for col in range(numCols):
                cell = rowCells[col]
                if cell[WALL_NORTH]:
                    diagram += ' _'
                else:
                    diagram += '  '
            diagram += "\n"

        lastCol = numCols - 1
        for col in range(numCols):
            cell = rowCells[col]

            if cell[WALL_WEST]:
                diagram += '|'
            else:
                diagram += ' '

            if cell[WALL_SOUTH]:
                diagram += '_'
            else:
                diagram += ' '

            if col == lastCol:
                if cell[WALL_EAST]:
                    diagram += '|'
                else:
                    diagram += ' '

        diagram += "\n"

    return diagram

def make_labyrinth(rows, cols, open_walls_chance=0):
    """
    Generate a labyrinth of the given size
    Returns a dictionary
    """

    def dfs_make(visited, labyrinth, row, col):
        def get_visited(visited, row, col):
            return visited[-(row + 1)][col]
        def set_visited(visited, row, col):
            visited[-(row + 1)][col] = True
        def get_unvisited_neighbors(row, col, max_row, max_col, visited):
            possible_neighbors = list()
            if row + 1 <= max_row and not get_visited(visited, row + 1, col):
                possible_neighbors.append((0, row + 1, col,)) # N
            if row - 1 >= 0 and not get_visited(visited, row - 1, col):
                possible_neighbors.append((1, row - 1, col,)) # S
            if col - 1 >= 0 and not get_visited(visited, row, col - 1):
                possible_neighbors.append((2, row, col - 1,)) # W
            if col + 1 <= max_col and not get_visited(visited, row, col + 1):
                possible_neighbors.append((3, row, col + 1,)) # E
            return possible_neighbors
        def open_walls(cells, row, col, neighbor):
            direction = neighbor[0]
            opposed = None
            if direction == 0:
                opposed = 1
            elif direction == 1:
                opposed = 0
            elif direction == 2:
                opposed = 3
            else:
                opposed = 2
            cells[-(row + 1)][col][direction] = False
            cells[-(neighbor[1] + 1)][neighbor[2]][opposed] = False
        def do_recurse(cells, row, col, neighbor, visited, labyrinth):
            open_walls(cells, row, col, neighbor)
            set_visited(visited, row, col)
            set_visited(visited, neighbor[1], neighbor[2])
            dfs_make(visited, labyrinth, neighbor[1], neighbor[2])
        cells = labyrinth["cells"]
        size = labyrinth["size"]
        max_row = size[0] - 1
        max_col = size[1] - 1
        while True:
            possible_neighbors = get_unvisited_neighbors(row, col, max_row, max_col, visited)
            neighbor = None
            if len(possible_neighbors) == 0:
                break
            else:
                neighbor = possible_neighbors[random.randint(0, len(possible_neighbors) - 1)]
                do_recurse(cells, row, col, neighbor, visited, labyrinth)

    labyrinth = dict()
    labyrinth["size"] = (rows, cols,)
    # set start position (on random edge)
    start_row = None
    start_col = None
    start_edge = random.randint(0, 3) # north, south, west, east
    if start_edge == 0:
        start_row = rows - 1
        start_col = random.randint(0, cols - 1)
    elif start_edge == 1:
        start_row = 0
        start_col = random.randint(0, cols - 1)
    elif start_edge == 2:
        start_col = 0
        start_row = random.randint(0, rows - 1)
    else:
        start_col = cols - 1
        start_row = random.randint(0, rows - 1)
    labyrinth["start"] = (start_row, start_col,)
    # initialize cells
    cells = list()
    visited = list()
    labyrinth["cells"] = cells
    for _ in range(0, rows):
        newrow = list()
        cells.append(newrow)
        visited_row = list()
        visited.append(visited_row)
        for _ in range(0, cols):
            newrow.append([True, True, True, True, ])
            visited_row.append(False)

    # generate random cells
    dfs_make(visited, labyrinth, start_row, start_col)

    # find end points
    labyrinth["ends"] = list()
    for nrow in range(0, rows):
        for ncol in range(0, cols):
            cell = cells[-(nrow + 1)][ncol]
            if (cell == [True, True, True, False,]
                or cell == [True, True, False, True,]
                or cell == [True, False, True, True,]
                or cell == [False, True, True, True,]
                ):
                labyrinth["ends"].append((nrow, ncol, ))

    # open random walls
    for nrow in range(1, rows - 1):
        for ncol in range(1, cols - 1):
            cell = cells[-(nrow + 1)][ncol]

            walls = list()
            for i in range(0, 4):
                if cell[i]:
                    walls.append(i) # north, south, west, east

            if len(walls) > 0 and random.randint(1, 100) <= open_walls_chance:
                cell_north = cells[-(nrow + 1 + 1)][ncol]
                cell_south = cells[-(nrow + 1 - 1)][ncol]
                cell_east = cells[-(nrow + 1)][ncol + 1]
                cell_west = cells[-(nrow + 1)][ncol - 1]

                direction = walls[random.randint(0, len(walls) - 1)]

                cell[direction] = False
                if direction == 0:
                    cell_north[WALL_SOUTH] = False
                elif direction == 1:
                    cell_south[WALL_NORTH] = False
                elif direction == 2:
                    cell_west[WALL_EAST] = False
                else:
                    cell_east[WALL_WEST] = False
    return labyrinth

def labyrinth_to_area(labyrinth):
    lab_size = labyrinth["size"]
    cells = labyrinth["cells"]
    area = []
    for _ in range(lab_size[1] * 2):
        area.append([])

    numRows = len(cells)
    for row in range(numRows):
        rowCells = cells[row]
        numCols = len(rowCells)
        for col in range(numCols):
            cell = rowCells[col]

            north = cell[WALL_NORTH]
            south = cell[WALL_SOUTH]
            west = cell[WALL_WEST]
            east = cell[WALL_EAST]

            areaRow1 = row * 2
            areaRow2 = row * 2 + 1

            cell_n = None
            if row > 0:
                cell_n = cells[row - 1][col]

            cell_s = None
            if row < numRows - 1:
                cell_s = cells[row + 1][col]

            cell_w = None
            if col > 0:
                cell_w = cells[row][col - 1]

            cell_e = None
            if col < numCols - 1:
                cell_e = cells[row][col + 1]

            # top left
            if north and west:
                area[areaRow1].append("<")
            elif north and not west:
                area[areaRow1].append("-")
            elif not north and west:
                area[areaRow1].append("(")
            elif not north and not west:
                if cell_n and (cell_n[WALL_SOUTH] or cell_n[WALL_WEST]):
                    area[areaRow1].append(";")
                elif cell_w and (cell_w[WALL_NORTH] or cell_w[WALL_EAST]):
                    area[areaRow1].append(";")
                else:
                    area[areaRow1].append("x")

            # top right
            if north and east:
                area[areaRow1].append(">")
            elif north and not east:
                area[areaRow1].append("-")
            elif not north and east:
                area[areaRow1].append(")")
            elif not north and not east:
                if cell_n and (cell_n[WALL_SOUTH] or cell_n[WALL_EAST]):
                    area[areaRow1].append(":")
                elif cell_e and (cell_e[WALL_NORTH] or cell_e[WALL_WEST]):
                    area[areaRow1].append(":")
                else:
                    area[areaRow1].append("x")

            # bottom left
            if south and west:
                area[areaRow2].append("[")
            elif south and not west:
                area[areaRow2].append("_")
            elif not south and west:
                area[areaRow2].append("(")
            elif not south and not west:
                if cell_s and (cell_s[WALL_NORTH] or cell_s[WALL_WEST]):
                    area[areaRow2].append(",")
                elif cell_w and (cell_w[WALL_SOUTH] or cell_w[WALL_EAST]):
                    area[areaRow2].append(",")
                else:
                    area[areaRow2].append("x")

            # bottom right
            if south and east:
                area[areaRow2].append("]")
            elif south and not east:
                area[areaRow2].append("_")
            elif not south and east:
                area[areaRow2].append(")")
            elif not south and not east:
                if cell_s and (cell_s[WALL_NORTH] or cell_s[WALL_EAST]):
                    area[areaRow2].append(".")
                elif cell_e and (cell_e[WALL_SOUTH] or cell_e[WALL_WEST]):
                    area[areaRow2].append(".")
                else:
                    area[areaRow2].append("x")

    return area

if __name__ == "__main__":
    import pprint
    pp = pprint.PrettyPrinter(indent=2)

    labyrinth = make_labyrinth(5, 5, 20)
    print(labyrinth_to_string(labyrinth))

    print("area:")
    area = labyrinth_to_area(labyrinth)
    pp.pprint(area)

    print("raw:")
    pp.pprint(labyrinth)
