from PlayerEnum import Player

class GameMovementRecorder:
    records = list()
    def record(self,player:Player,old_player,new_player,defeateds):
        self.records.append([player,old_player,new_player,defeateds])

    def show(self):
        for movement in self.records:
            print(movement)