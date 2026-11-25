"""
Canonical COCO-80 and Open-Vocabulary class ID mappings for IMGSTD & VIDSTD.
Deterministic 4-digit IDs aligned with spec/vocabulary.json.
"""

from typing import Dict, Tuple

COCO_CLASS_ID_MAP: Dict[int, str] = {
    0: "0001",   # person
    1: "0002",   # bicycle
    2: "0003",   # car
    3: "0004",   # motorcycle
    4: "0005",   # airplane
    5: "0006",   # bus
    6: "0007",   # train
    7: "0008",   # truck
    8: "0009",   # boat
    9: "0010",   # traffic light
    10: "0011",  # fire hydrant
    11: "0012",  # stop sign
    12: "0013",  # parking meter
    13: "0014",  # bench
    14: "0016",  # bird
    15: "0017",  # cat
    16: "0018",  # dog
    17: "0019",  # horse
    18: "0020",  # sheep
    19: "0021",  # cow
    20: "0022",  # elephant
    21: "0023",  # bear
    22: "0024",  # zebra
    23: "0025",  # giraffe
    24: "0026",  # backpack
    25: "0027",  # umbrella
    26: "0028",  # handbag
    27: "0029",  # tie
    28: "0030",  # suitcase
    29: "0031",  # frisbee
    30: "0032",  # skis
    31: "0033",  # snowboard
    32: "0034",  # sports ball
    33: "0035",  # kite
    34: "0036",  # baseball bat
    35: "0037",  # baseball glove
    36: "0038",  # skateboard
    37: "0039",  # surfboard
    38: "0040",  # tennis racket
    39: "0041",  # bottle
    40: "0042",  # wine glass
    41: "0043",  # cup
    42: "0044",  # fork
    43: "0045",  # knife
    44: "0046",  # spoon
    45: "0047",  # bowl
    46: "0048",  # banana
    47: "0049",  # apple
    48: "0050",  # sandwich
    49: "0051",  # orange
    50: "0052",  # broccoli
    51: "0053",  # carrot
    52: "0054",  # hot dog
    53: "0055",  # pizza
    54: "0056",  # donut
    55: "0082",  # cake
    56: "0057",  # chair
    57: "0058",  # couch
    58: "0059",  # potted plant
    59: "0060",  # bed
    60: "0061",  # dining table
    61: "0062",  # toilet
    62: "0064",  # tv
    63: "0063",  # laptop
    64: "0065",  # mouse
    65: "0066",  # remote
    66: "0067",  # keyboard
    67: "0068",  # cell phone
    68: "0069",  # microwave
    69: "0070",  # oven
    70: "0071",  # toaster
    71: "0072",  # sink
    72: "0073",  # refrigerator
    73: "0074",  # book
    74: "0075",  # clock
    75: "0076",  # vase
    76: "0077",  # scissors
    77: "0078",  # teddy bear
    78: "0079",  # hair drier
    79: "0080",  # toothbrush
}

OPEN_VOCAB_MAP: Dict[str, Tuple[str, str, str]] = {
    # Celestial & Space Bodies
    "planet earth": ("0201", "Planet Earth / Globe (Celestial)", "Celestial & Space"),
    "earth": ("0201", "Planet Earth / Globe (Celestial)", "Celestial & Space"),
    "globe": ("0201", "Planet Earth / Globe (Celestial)", "Celestial & Space"),
    "planet": ("0202", "Planet / Exoplanet (Celestial)", "Celestial & Space"),
    "moon": ("0203", "Moon / Lunar Body (Celestial)", "Celestial & Space"),
    "star": ("0204", "Sun / Star (Celestial)", "Celestial & Space"),
    "sun": ("0204", "Sun / Star (Celestial)", "Celestial & Space"),
    "galaxy": ("0205", "Galaxy / Milky Way (Celestial)", "Celestial & Space"),
    "outer space": ("0206", "Outer Space / Deep Space (Celestial)", "Celestial & Space"),
    "space": ("0206", "Outer Space / Deep Space (Celestial)", "Celestial & Space"),
    "satellite": ("0207", "Satellite / Spacecraft (Spacecraft)", "Spacecraft"),
    "nebula": ("0208", "Nebula / Cosmic Cloud (Celestial)", "Celestial & Space"),
    "atmosphere": ("0209", "Atmosphere / Planetary Limb (Atmosphere)", "Atmosphere"),
    "continent": ("0210", "Continent / Landmass (Geography)", "Geography"),
    "ocean": ("0211", "Ocean / Sea (Geography)", "Geography"),

    # Architecture & Environment
    "building facade": ("0101", "Building / Storefront Facade", "Architecture"),
    "building": ("0101", "Building / Storefront Facade", "Architecture"),
    "storefront": ("0101", "Building / Storefront Facade", "Architecture"),
    "glass door": ("0102", "Door / Entrance", "Architecture"),
    "door": ("0102", "Door / Entrance", "Architecture"),
    "window": ("0103", "Window / Glazing", "Architecture"),
    "stairs": ("0104", "Stairs / Steps", "Architecture"),
    "signboard": ("0105", "Signboard / Banner", "Architecture"),
    "sign": ("0105", "Signboard / Banner", "Architecture"),
    "wall": ("0106", "Wall / Partition", "Architecture"),
    "text banner": ("0105", "Signboard / Banner", "Architecture"),
    "room": ("0100", "Architectural Structure / Facade", "Architecture"),
    "street": ("0905", "Ground / Floor / Surface", "Background"),
    "sky": ("0904", "Sky / Upper Backdrop", "Background"),

    # Documents, UI & Media
    "screen": ("0120", "UI Screen / Display Area", "User Interface"),
    "chart": ("0121", "UI Graphical Chart / Diagram", "User Interface"),
    "diagram": ("0121", "UI Graphical Chart / Diagram", "User Interface"),
    "logo": ("0123", "UI Icon / Glyph", "User Interface"),

    # Natural / Outdoor / Industrial Open-Vocabulary
    "desert": ("0212", "Desert / Sand Landscape", "Geography"),
    "sand dune": ("0213", "Sand Dune (Geography)", "Geography"),
    "sand": ("0213", "Desert Sand (Geography)", "Geography"),
    "heavy machinery": ("0214", "Heavy Industrial Machinery", "Machinery"),
    "industrial machine": ("0214", "Heavy Industrial Machinery", "Machinery"),
    "spaceship": ("0215", "Spacecraft / Futuristic Vehicle", "Vehicle"),
    "aircraft": ("0005", "Airplane (Transport)", "Vehicle"),
    "helicopter": ("0216", "Helicopter / Flyer", "Vehicle"),
    "rodent": ("0217", "Small Rodent / Mouse", "Animal"),
    "soldier": ("0218", "Soldier / Armored Figure", "Human"),
    "animal": ("0219", "Animal / Wildlife", "Animal"),
    "rock": ("0220", "Rock / Cliff Formation", "Geography"),
    "mountain": ("0220", "Mountain / Rocky Terrain", "Geography"),

    # Standard Physical Objects (aligned with COCO_CLASS_ID_MAP)
    "person": ("0001", "Person (Human)", "Human"),
    "bicycle": ("0002", "Bicycle", "Vehicle"),
    "car": ("0003", "Car", "Vehicle"),
    "motorcycle": ("0004", "Motorcycle", "Vehicle"),
    "airplane": ("0005", "Airplane", "Vehicle"),
    "bus": ("0006", "Bus", "Vehicle"),
    "train": ("0007", "Train", "Vehicle"),
    "truck": ("0008", "Truck", "Vehicle"),
    "boat": ("0009", "Boat / Ship", "Vehicle"),
    "traffic light": ("0010", "Traffic Light (Infrastructure)", "Infrastructure"),
    "bench": ("0014", "Bench", "Object"),
    "bird": ("0016", "Bird (Animal)", "Animal"),
    "cat": ("0017", "Cat (Animal)", "Animal"),
    "dog": ("0018", "Dog (Animal)", "Animal"),
    "chair": ("0057", "Chair", "Furniture"),
    "couch": ("0058", "Couch / Sofa", "Furniture"),
    "potted plant": ("0059", "Potted Plant", "Object"),
    "bed": ("0060", "Bed", "Furniture"),
    "dining table": ("0061", "Dining Table", "Furniture"),
    "cake": ("0082", "Cake (Food)", "Food"),
    "laptop": ("0063", "Laptop", "Electronics"),
    "mouse": ("0065", "Mouse", "Object"),
    "remote": ("0066", "Remote", "Electronics"),
    "keyboard": ("0067", "Keyboard", "Electronics"),
    "cell phone": ("0068", "Cell Phone", "Electronics"),
    "book": ("0074", "Book", "Object"),
    "clock": ("0075", "Clock", "Object"),
}
