"""North America 2025 component data; sources in docs/rules-ticket-to-ride.md."""

from typing import NamedTuple


class Route(NamedTuple):
    cities: tuple[int, int]
    length: int
    color: int


class Ticket(NamedTuple):
    cities: tuple[int, int]
    points: int


CITIES = (
    "Atlanta",
    "Boston",
    "Calgary",
    "Charleston",
    "Chicago",
    "Dallas",
    "Denver",
    "Duluth",
    "El Paso",
    "Helena",
    "Houston",
    "Kansas City",
    "Las Vegas",
    "Little Rock",
    "Los Angeles",
    "Miami",
    "Montreal",
    "Nashville",
    "New Orleans",
    "New York",
    "Oklahoma City",
    "Omaha",
    "Phoenix",
    "Pittsburgh",
    "Portland",
    "Raleigh",
    "Saint Louis",
    "Salt Lake City",
    "San Francisco",
    "Santa Fe",
    "Sault St. Marie",
    "Seattle",
    "Toronto",
    "Vancouver",
    "Washington",
    "Winnipeg",
)
CITY_INDEX = {name: i for i, name in enumerate(CITIES)}

ROUTES = (
    Route((33, 2), 3, -1),  # Vancouver - Calgary
    Route((33, 31), 1, -1),  # Vancouver - Seattle
    Route((33, 31), 1, -1),  # Vancouver - Seattle
    Route((31, 2), 4, -1),  # Seattle - Calgary
    Route((31, 9), 6, 3),  # Seattle - Helena
    Route((31, 24), 1, -1),  # Seattle - Portland
    Route((31, 24), 1, -1),  # Seattle - Portland
    Route((24, 27), 6, 5),  # Portland - Salt Lake City
    Route((24, 28), 5, 4),  # Portland - San Francisco
    Route((24, 28), 5, 0),  # Portland - San Francisco
    Route((28, 27), 5, 2),  # San Francisco - Salt Lake City
    Route((28, 27), 5, 6),  # San Francisco - Salt Lake City
    Route((28, 14), 3, 3),  # San Francisco - Los Angeles
    Route((28, 14), 3, 0),  # San Francisco - Los Angeles
    Route((14, 12), 2, -1),  # Los Angeles - Las Vegas
    Route((14, 22), 3, -1),  # Los Angeles - Phoenix
    Route((14, 8), 6, 7),  # Los Angeles - El Paso
    Route((2, 35), 6, 6),  # Calgary - Winnipeg
    Route((2, 9), 4, -1),  # Calgary - Helena
    Route((9, 35), 4, 5),  # Helena - Winnipeg
    Route((9, 27), 3, 0),  # Helena - Salt Lake City
    Route((9, 6), 4, 4),  # Helena - Denver
    Route((9, 7), 6, 2),  # Helena - Duluth
    Route((9, 21), 5, 1),  # Helena - Omaha
    Route((27, 6), 3, 1),  # Salt Lake City - Denver
    Route((27, 6), 3, 3),  # Salt Lake City - Denver
    Route((12, 27), 3, 2),  # Las Vegas - Salt Lake City
    Route((22, 6), 5, 6),  # Phoenix - Denver
    Route((22, 29), 3, -1),  # Phoenix - Santa Fe
    Route((22, 8), 3, -1),  # Phoenix - El Paso
    Route((35, 30), 6, -1),  # Winnipeg - Sault St. Marie
    Route((35, 7), 4, 7),  # Winnipeg - Duluth
    Route((7, 30), 3, -1),  # Duluth - Sault St. Marie
    Route((7, 32), 6, 0),  # Duluth - Toronto
    Route((7, 4), 3, 1),  # Duluth - Chicago
    Route((7, 21), 2, -1),  # Duluth - Omaha
    Route((7, 21), 2, -1),  # Duluth - Omaha
    Route((21, 4), 4, 5),  # Omaha - Chicago
    Route((21, 11), 1, -1),  # Omaha - Kansas City
    Route((21, 11), 1, -1),  # Omaha - Kansas City
    Route((11, 26), 2, 5),  # Kansas City - Saint Louis
    Route((11, 26), 2, 0),  # Kansas City - Saint Louis
    Route((11, 20), 2, -1),  # Kansas City - Oklahoma City
    Route((11, 20), 2, -1),  # Kansas City - Oklahoma City
    Route((20, 13), 2, -1),  # Oklahoma City - Little Rock
    Route((20, 5), 2, -1),  # Oklahoma City - Dallas
    Route((20, 5), 2, -1),  # Oklahoma City - Dallas
    Route((5, 13), 2, -1),  # Dallas - Little Rock
    Route((5, 10), 1, -1),  # Dallas - Houston
    Route((5, 10), 1, -1),  # Dallas - Houston
    Route((10, 18), 2, -1),  # Houston - New Orleans
    Route((8, 10), 6, 4),  # El Paso - Houston
    Route((8, 5), 4, 1),  # El Paso - Dallas
    Route((8, 20), 5, 3),  # El Paso - Oklahoma City
    Route((8, 29), 2, -1),  # El Paso - Santa Fe
    Route((29, 20), 3, 5),  # Santa Fe - Oklahoma City
    Route((20, 6), 4, 1),  # Oklahoma City - Denver
    Route((29, 6), 2, -1),  # Santa Fe - Denver
    Route((6, 11), 4, 7),  # Denver - Kansas City
    Route((6, 11), 4, 2),  # Denver - Kansas City
    Route((6, 21), 4, 0),  # Denver - Omaha
    Route((18, 15), 6, 1),  # New Orleans - Miami
    Route((18, 0), 4, 2),  # New Orleans - Atlanta
    Route((18, 0), 4, 3),  # New Orleans - Atlanta
    Route((18, 13), 3, 4),  # New Orleans - Little Rock
    Route((13, 17), 3, 6),  # Little Rock - Nashville
    Route((13, 26), 2, -1),  # Little Rock - Saint Louis
    Route((26, 17), 2, -1),  # Saint Louis - Nashville
    Route((26, 23), 5, 4),  # Saint Louis - Pittsburgh
    Route((26, 4), 2, 4),  # Saint Louis - Chicago
    Route((26, 4), 2, 6),  # Saint Louis - Chicago
    Route((4, 23), 3, 7),  # Chicago - Pittsburgh
    Route((4, 23), 3, 2),  # Chicago - Pittsburgh
    Route((4, 32), 4, 6),  # Chicago - Toronto
    Route((30, 16), 5, 7),  # Sault St. Marie - Montreal
    Route((32, 16), 3, -1),  # Toronto - Montreal
    Route((30, 32), 2, -1),  # Sault St. Marie - Toronto
    Route((32, 23), 2, -1),  # Toronto - Pittsburgh
    Route((23, 19), 2, 6),  # Pittsburgh - New York
    Route((23, 19), 2, 4),  # Pittsburgh - New York
    Route((23, 34), 2, -1),  # Pittsburgh - Washington
    Route((23, 25), 2, -1),  # Pittsburgh - Raleigh
    Route((17, 25), 3, 7),  # Nashville - Raleigh
    Route((17, 0), 1, -1),  # Nashville - Atlanta
    Route((17, 23), 4, 3),  # Nashville - Pittsburgh
    Route((0, 15), 5, 5),  # Atlanta - Miami
    Route((0, 3), 2, -1),  # Atlanta - Charleston
    Route((0, 25), 2, -1),  # Atlanta - Raleigh
    Route((0, 25), 2, -1),  # Atlanta - Raleigh
    Route((3, 15), 4, 0),  # Charleston - Miami
    Route((25, 3), 2, -1),  # Raleigh - Charleston
    Route((25, 34), 2, -1),  # Raleigh - Washington
    Route((25, 34), 2, -1),  # Raleigh - Washington
    Route((34, 19), 2, 2),  # Washington - New York
    Route((34, 19), 2, 7),  # Washington - New York
    Route((19, 1), 2, 3),  # New York - Boston
    Route((19, 1), 2, 1),  # New York - Boston
    Route((19, 16), 3, 5),  # New York - Montreal
    Route((1, 16), 2, -1),  # Boston - Montreal
    Route((1, 16), 2, -1),  # Boston - Montreal
)

TICKETS = (
    Ticket((14, 19), 21),  # Los Angeles - New York
    Ticket((7, 10), 8),  # Duluth - Houston
    Ticket((30, 17), 8),  # Sault St. Marie - Nashville
    Ticket((19, 0), 6),  # New York - Atlanta
    Ticket((24, 17), 17),  # Portland - Nashville
    Ticket((33, 16), 20),  # Vancouver - Montreal
    Ticket((7, 8), 10),  # Duluth - El Paso
    Ticket((32, 15), 10),  # Toronto - Miami
    Ticket((24, 22), 11),  # Portland - Phoenix
    Ticket((5, 19), 11),  # Dallas - New York
    Ticket((2, 27), 7),  # Calgary - Salt Lake City
    Ticket((2, 22), 13),  # Calgary - Phoenix
    Ticket((14, 15), 20),  # Los Angeles - Miami
    Ticket((35, 13), 11),  # Winnipeg - Little Rock
    Ticket((28, 0), 17),  # San Francisco - Atlanta
    Ticket((11, 10), 5),  # Kansas City - Houston
    Ticket((14, 4), 16),  # Los Angeles - Chicago
    Ticket((6, 23), 11),  # Denver - Pittsburgh
    Ticket((4, 29), 9),  # Chicago - Santa Fe
    Ticket((33, 29), 13),  # Vancouver - Santa Fe
    Ticket((1, 15), 12),  # Boston - Miami
    Ticket((4, 18), 7),  # Chicago - New Orleans
    Ticket((16, 0), 9),  # Montreal - Atlanta
    Ticket((31, 19), 22),  # Seattle - New York
    Ticket((6, 8), 4),  # Denver - El Paso
    Ticket((9, 14), 8),  # Helena - Los Angeles
    Ticket((35, 10), 12),  # Winnipeg - Houston
    Ticket((16, 18), 13),  # Montreal - New Orleans
    Ticket((30, 20), 9),  # Sault St. Marie - Oklahoma City
    Ticket((31, 14), 9),  # Seattle - Los Angeles
    Ticket((0, 34), 4),  # Atlanta - Washington
    Ticket((21, 18), 8),  # Omaha - New Orleans
    Ticket((28, 34), 21),  # San Francisco - Washington
)
