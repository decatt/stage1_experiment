import numpy


class ScriptUnit:
    def __init__(self, pos, unit_type):
        self.state = 0
        self.target = -1
        self.x = pos[0]
        self.y = pos[1]
        self.unit_type = unit_type

    def action(self, obs, vu):
        res = []
        if obs[0][self.y][self.x][self.unit_type] == 1 and vu.count(16*self.y+self.x) == 1:
            if self.state == 0:
                self.state = self.select_state()
            if self.target == -1:
                self.target = self.select_target()
            if self.state == 1:
                res = self.produce(obs)
            if self.state == 2:
                res = self.move(obs)
            if self.state == 3:
                res = self.harvest(obs)
            if self.state == 4:
                res = self.attack()
        else:
            res = [34, 0, 0, 0, 0, 0, 0, 0]
        return res

    def select_state(self):
        return 0

    def select_target(self):
        if self.state == 3:
            return 16
        return -1

    def produce(self, obs):
        res = []
        tx = self.target % 16
        ty = self.target // 16
        if self.unit_type == 15:
            if self.x == tx + 1 and self.y == ty and obs[0][ty][tx][13] == 1:
                res = [self.x + 16 * self.y, 4, 0, 0, 0, 0, 3, 3, 0]
            elif self.x == tx - 1 and self.y == ty and obs[0][ty][tx][13] == 1:
                res = [self.x + 16 * self.y, 4, 0, 0, 0, 0, 1, 3, 0]
            elif self.y == ty - 1 and self.x == tx and obs[0][ty][tx][13] == 1:
                res = [self.x + 16 * self.y, 4, 0, 0, 0, 0, 2, 3, 0]
            elif self.y == ty + 1 and self.x == tx and obs[0][ty][tx][13] == 1:
                res = [self.x + 16 * self.y, 4, 0, 0, 0, 0, 0, 3, 0]
            else:
                res = [34, 0, 0, 0, 0, 0, 0, 0]
        else:
            res = [34, 0, 0, 0, 0, 0, 0, 0]
        if res is not [34, 0, 0, 0, 0, 0, 0, 0]:
            self.state = 0
        return res

    def move(self, obs):
        res = []
        tx = self.target % 16
        ty = self.target//16
        if self.x > tx and obs[0][self.y][self.x-1][13] == 1:
            res = [self.x+16*self.y, 1, 3, 0, 0, 0, 0, 0, 0]
            self.x = self.x-1
        elif self.x < tx and obs[0][self.y][self.x+1][13] == 1:
            res = [self.x+16*self.y, 1, 1, 0, 0, 0, 0, 0, 0]
            self.x = self.x + 1
        elif self.y < ty and obs[0][self.y+1][self.x][13] == 1:
            res = [self.x+16*self.y, 1, 2, 0, 0, 0, 0, 0, 0]
            self.y = self.y + 1
        elif self.y > ty and obs[0][self.y-1][self.x][13] == 1:
            res = [self.x+16*self.y, 1, 0, 0, 0, 0, 0, 0, 0]
            self.y = self.y-1
        else:
            res = [34,0,0,0,0,0,0,0]
        return res

    def harvest(self, obs):
        if obs[0][self.y][self.x][5] == 1:
            self.target = 16
            if obs[0][self.y-1][self.x][14] == 1:
                return [self.x + 16 * self.y, 2, 0, 0, 0, 0, 0, 0, 0]
            elif obs[0][self.y][self.x+1][14] == 1:
                return [self.x + 16 * self.y, 2, 0, 1, 0, 0, 0, 0, 0]
            elif obs[0][self.y+1][self.x][14] == 1:
                return [self.x + 16 * self.y, 2, 0, 2, 0, 0, 0, 0, 0]
            elif obs[0][self.y][self.x-1][14] == 1:
                return [self.x + 16 * self.y, 2, 0, 3, 0, 0, 0, 0, 0]
            else:
                return self.move(obs)
        else:
            self.target = 34
            if obs[0][self.y-1][self.x][15] == 1:
                return [self.x + 16 * self.y, 3, 0, 0, 0, 0, 0, 0, 0]
            elif obs[0][self.y][self.x+1][15] == 1:
                return [self.x + 16 * self.y, 3, 0, 0, 1, 0, 0, 0, 0]
            elif obs[0][self.y+1][self.x][15] == 1:
                return [self.x + 16 * self.y, 3, 0, 0, 2, 0, 0, 0, 0]
            elif obs[0][self.y][self.x-1][15] == 1:
                return [self.x + 16 * self.y, 3, 0, 0, 3, 0, 0, 0, 0]
            else:
                return self.move(obs)

    def attack(self):
        return [self.x+16*self.y, 1, self.target, 0, 0, 0, 0, 0, 0]
