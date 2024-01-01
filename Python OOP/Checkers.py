import cv2 as cv
from GlobalFunctions import *
from MovementEnum import Movement
from PlayerEnum import Player

class Checkers():
    square_index_with_white_checkers_old = list()
    square_index_with_white_checkers_new = list()
    square_index_with_black_checkers_old = list()
    square_index_with_black_checkers_new = list()
    movement = None
    movement_old_index = None
    movement_new_index = None
    defeated_indexes = None
    def get_indexes_changes(self,old_indexes,new_indexes):
        """
            Return the indexes that old_indexes have but new_indexes doesn't
        """
        result = list()
        for old in old_indexes:
            flag = False
            for new in new_indexes:
                if old == new:
                    flag = True
                    break
            if not flag:
                result.append(old)
        return result
    def what_happaned(self,nnw,now,nnb,nob):
        """
            Decide what kind of movement has happened
            nnw = number of new white checkers
            now = number of old white checkers
            nnb = number of new black checkers
            nob = number of old black checkers
        """
        if nnw == 1 and nnb == 0:
            if nob > 0:
                return Movement.WhiteWithBlack
            return Movement.White
        
        elif nnb == 1 and nnw == 0:
            if now > 0:
                return Movement.BlackWithWhite
            return Movement.Black
        
        return Movement.Wrong

    def find_different_index(self): 
        # Eskide olup da yenide olmayan 
        old_sqaure_white_indexes = self.get_indexes_changes(self.square_index_with_white_checkers_old,self.square_index_with_white_checkers_new)
        old_sqaure_black_indexes = self.get_indexes_changes(self.square_index_with_black_checkers_old,self.square_index_with_black_checkers_new)
        
        # Yenide olup da eskide olmayan 
        new_sqaure_white_indexes = self.get_indexes_changes(self.square_index_with_white_checkers_new,self.square_index_with_white_checkers_old)
        new_sqaure_black_indexes = self.get_indexes_changes(self.square_index_with_black_checkers_new,self.square_index_with_black_checkers_old)
        number_new_white = len(new_sqaure_white_indexes) 
        number_old_white = len(old_sqaure_white_indexes) 
        number_new_black = len(new_sqaure_black_indexes) 
        number_old_black = len(old_sqaure_black_indexes)   

        self.movement = self.what_happaned(number_new_white,number_old_white,number_new_black,number_old_black)      

        player = self.get_player(self.movement)
        self.defeated_indexes = None

        match self.movement:
            case Movement.White:
                self.movement_old_index = self.get_closest_index(old_sqaure_white_indexes,new_sqaure_white_indexes[0])
                self.movement_new_index = new_sqaure_white_indexes[0]
            case Movement.WhiteWithBlack:
                self.movement_old_index = self.get_closest_index(old_sqaure_white_indexes,new_sqaure_white_indexes[0])
                self.movement_new_index = new_sqaure_white_indexes[0]
                self.defeated_indexes = old_sqaure_black_indexes
            case Movement.Black:
                self.movement_old_index = self.get_closest_index(old_sqaure_black_indexes,new_sqaure_black_indexes[0])
                self.movement_new_index = new_sqaure_black_indexes[0]            
            case Movement.BlackWithWhite:
                self.movement_old_index = self.get_closest_index(old_sqaure_black_indexes,new_sqaure_black_indexes[0])
                self.movement_new_index = new_sqaure_black_indexes[0]
                self.defeated_indexes = old_sqaure_white_indexes                
            case _:
                self.movement_old_index = None
                self.movement_new_index = None

        if self.movement != Movement.Wrong:
            return (player,self.movement_old_index,self.movement_new_index,self.defeated_indexes)
        
        return None
    
    def get_player(self,movement):
        match movement:
            case Movement.White:
                return Player.White
            case Movement.WhiteWithBlack:
                return Player.White
            case Movement.Black:
                return Player.Black
            case Movement.BlackWithWhite:
                return Player.Black

    def set_square_white_index(self,new_index:list):
        self.square_index_with_white_checkers_old = self.square_index_with_white_checkers_new
        self.square_index_with_white_checkers_new = new_index
    def set_square_black_index(self,new_index:list):
        self.square_index_with_black_checkers_old = self.square_index_with_black_checkers_new
        self.square_index_with_black_checkers_new = new_index
        
    def get_closest_index(self,indexes,point):
        if len(indexes) == 0:
            return 
        
        result = indexes[0]
        distance_min = abs(indexes[0][0] - point[0]) + abs(indexes[0][1]-point[1])
        for index in indexes:
            distance = abs(index[0] - point[0]) + abs(index[1]-point[1])
            if distance < distance_min:
                result = index
                distance_min = distance
        return result
    
    def apply_changes(self,bird_eye_frame,radius):
        if self.movement_new_index == None or self.movement_old_index == None:
            return
        
        mover_old_position = get_square_position(self.movement_old_index)
        mover_new_position = get_square_position(self.movement_new_index)
        cv.putText(bird_eye_frame,f"Old {self.movement_old_index}",(mover_old_position[0]-10,mover_old_position[1]-20),1,1,(255,0,0),2)
        cv.circle(bird_eye_frame, (mover_old_position[0], mover_old_position[1]), radius, (255, 0, 0), 3)
        cv.putText(bird_eye_frame,f"New {self.movement_new_index}",(mover_new_position[0]-10,mover_new_position[1]-20),1,1,(0,255,0),2)
        cv.circle(bird_eye_frame, (mover_new_position[0], mover_new_position[1]), radius, (0, 255, 0), 3)
        cv.arrowedLine(bird_eye_frame,mover_old_position,mover_new_position,(255,255,0),2)
        if self.defeated_indexes != None: 
            defeated_positions = [get_square_position(x) for x in self.defeated_indexes]
            for cent,position in zip(self.defeated_indexes,defeated_positions):
                cv.putText(bird_eye_frame,f"Defeated {cent}",(position[0]-10,position[1]-20),1,1,(0,0,255),2)
                cv.circle(bird_eye_frame, (position[0], position[1]), radius, (0, 0, 255), 3)    

