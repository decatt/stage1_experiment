class MapParameter:
    def __init__(self, map_h=16, map_w=16):
        self.map_h = map_h,
        self.map_w = map_w,
        self.action_types = {'noop': 0, 'move': 1, 'harvest': 2, 'return': 3, 'produce': 4, 'attack': 5}
        self.noop = 0,
        self.move = 1,
        self.harvest = 2,
        self.action_return = 3,
        self.produce = 4,
        self.attack = 5,
        self.produce_unit_types = {'resource': 0, 'base': 1, 'barrack': 2, 'worker': 3, 'light': 4, 'heavy': 5,
                                   'ranged': 6},
        self.p_resource = 0,
        self.p_base = 1,
        self.p_barrack = 2,
        self.p_worker = 3,
        self.p_light = 4,
        self.p_heavy = 5,
        self.p_ranged = 6,
        self.direction = {'north': 0, 'east': 1, 'south': 2, 'west': 3},
        self.north = 0,
        self.east = 1,
        self.south = 2,
        self.west = 3


