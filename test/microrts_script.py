import numpy


class ScriptUnit:
    def __init__(self, pos, unit_type):
        self.state = 0
        self.target = -1
        self.x = pos[0]
        self.y = pos[1]
        self.type = unit_type

    def action(self, obs):
        res = []
        if self.state == 0:
            self.state = self.select_state()
        if self.target == -1:
            self.target = self.select_target()
        if self.state == 1:
            res = self.produce(obs)
        if self.state == 2:
            res = self.move()
        if self.state == 3:
            res = self.harvest()
        if self.state == 4:
            res = self.attack()
        return res

    def select_state(self):
        return 0

    def select_target(self):
        return 0

    def produce(self, obs):
        action = [self.x+16*self.y, 4, 0, 0, 0, 0, self.target, 0, 0]
        return action

    def move(self):
        return [self.x+16*self.y, 1, self.target, 0, 0, 0, 0, 0, 0]

    def harvest(self, obs):
        if obs[0][self.y][self.x][6] == 1:
            if obs[0][self.y-1][self.x][14] == 1:
                return [self.x + 16 * self.y, 2, 0, 0, 0, 0, 0, 0, 0]
            elif obs[0][self.y][self.x+1][14] == 1:
                return [self.x + 16 * self.y, 2, 0, 1, 0, 0, 0, 0, 0]
            elif obs[0][self.y+1][self.x][14] == 1:
                return [self.x + 16 * self.y, 2, 0, 2, 0, 0, 0, 0, 0]
            elif obs[0][self.y][self.x-1][14] == 1:
                return [self.x + 16 * self.y, 2, 0, 3, 0, 0, 0, 0, 0]
            else:
                self.move()
        else:
            self.target = 0
            self.move()

    def attack(self):
        return [self.x+16*self.y, 1, self.target, 0, 0, 0, 0, 0, 0]
