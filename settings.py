WIDTH = 980
HEIGHT = 675
TITLE_STRING = 'Sushi Go!'
FPS = 60

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

AZUL = (52, 152, 219)
ROJO = (230, 25, 25)
ROSA = (216, 27, 96)
AMARILLO = (241, 196, 15)
NARANJA = (220, 118, 51)
COLORS = (AZUL, ROJO, ROSA, AMARILLO, NARANJA)

VERDE = (111, 255, 77) # selection green
GREY = (46, 46, 46) # Arrow grey

P2 = 2
P3 = 3
P4 = 4
P5 = 5

from Point import *
class AppConfig:
    def __init__(self, num_players=5):
        # Path of card images
        self.card_path = './img/cards' 
        # Path of the font (used in all texts)
        self.FONT_PATH = './fonts/AtomicMd.otf' 
        # Hand/Player text
        self.FONT_SIZE = {P2: 20, P3: 20, P4: 40, P5: 40}[num_players]
        # Horizontal distance between Played and Hand (centered)
        self.desp = 40

        # CARDS
        # Cards initial position
        self.POS_INI = {P2: Point(x = 75 + self.desp, y = 290), 
                        P3: Point(x = 80 + self.desp, y = 200), 
                        P4: Point(x = 165 + self.desp, y = 35), 
                        P5: Point(x = 250 + self.desp, y = 35)} [num_players]
        # Card dimensions
        self.CARD_DIM = Point(x = 81, y = 99)
        # Distance between cards
        self.CARD_DIST = {P2: Point(x = 0, y = 30), 
                          P3: Point(x = 10, y = 20), 
                          P4: Point(x = 10, y = 30), 
                          P5: Point(x = 10, y = 8)} [num_players]
        
        # Hand/Played Text Rectangle (little semi-transparent)
        self.POS_TEXT_INI = {P2: Point(15 + self.desp, self.POS_INI.y), 
                             P3: Point(20 + self.desp, self.POS_INI.y), 
                             P4: Point(20 + self.desp, self.POS_INI.y), 
                             P5: Point(20 + self.desp, self.POS_INI.y)}[num_players]
        # Hand/Played Text Rectange (little semi-transparent)
        self.TEXT_W = self.POS_INI.x - self.POS_TEXT_INI.x - self.CARD_DIST.x
        self.TEXT_H = self.CARD_DIM.y
        
        # Arrows dimensions (previous: 15, 20, 20, 20)
        self.BUTTON_DIM = {P2: Point(x = 15, y = self.CARD_DIM.y), 
                           P3: Point(x = 15, y = self.CARD_DIM.y), 
                           P4: Point(x = 15, y = self.CARD_DIM.y), 
                           P5: Point(x = 15, y = self.CARD_DIM.y)}[num_players]
        self.ARROW_SCALE = 0.5
        
        self.logo_scale = {P2: 0.5,  P3: 0.35, 
                           P4: 0,    P5: 0}[num_players]
        
        # Little Pudin text size
        self.PUDIN_FONT_SIZE = 15
        # Size of little pudin image
        self.l_pudin_scale = 0.9
        # Vertical distance down (from played text to little img)
        self.l_pudin_h = {P2: 20, 
                          P3: 20, 
                          P4: 20, 
                          P5: 28}[num_players]
        # Adjusting number height
        self.diff_text_puddin = 2
        # Horizontal distance between img and x[Num]
        self.dist_text_puddin = {P2: 24, 
                                 P3: 20, 
                                 P4: 24, 
                                 P5: 24}[num_players]
        
        # Round font size (in the top-left of the window)
        self.ROUND_FONT_SIZE = {P2: 30, 
                                P3: 30, 
                                P4: 15, 
                                P5: 18}[num_players]
        # Distance from Score 'topright' to the top-right of the screen
        self.ROUND_X = 15
        self.ROUND_Y = 8
        # Black text-background
        self.ROUND_BG_MARGIN = {P2: 10, 
                                P3: 10, 
                                P4: 5, 
                                P5: 5}[num_players]
        # "Sushi Go!" message position (in the center of hand)
        self.SUSHIGO_MESSAGE_POS = {P2: Point(WIDTH//2, self.POS_INI.y + self.CARD_DIM.y//2 + 4), 
                                    P3: Point(WIDTH//2 , self.POS_INI.y + self.CARD_DIM.y // 2 + 4), 
                                    P4: Point(WIDTH//2, 5 + self.POS_INI.y + self.CARD_DIM.y // 2), 
                                    P5: Point(WIDTH//2, 5 + self.POS_INI.y + self.CARD_DIM.y // 2)}[num_players]
        # "Listo" button size
        self.LISTO_SCALE = 0.6
        # "Listo" button center position
        self.LISTO_POS = {P2: Point(932, self.POS_INI.y + self.CARD_DIM.y + self.CARD_DIST.y + self.CARD_DIM.y//2), 
                          P3: Point(936, self.POS_INI.y + self.CARD_DIM.y + self.CARD_DIST.y + self.CARD_DIM.y//2), 
                          P4: Point(930, HEIGHT // 2 + 12), 
                          P5: Point(930, HEIGHT // 2 - self.CARD_DIM.y // 2 + 8)}[num_players]
        

        # RENDER_SCORES
        # player text rectangles
        self.SCORE_PLAYER_DIM = {
            P2: Point(x = 146, y = 70), 
            P3: Point(x = 128, y = 61), 
            P4: Point(x = 146, y = 70), 
            P5: Point(x = 146, y = 70)}[num_players]
        # score number rectangles 
        self.SCORE_NUMBER_DIM = {
            P2: Point(x = self.SCORE_PLAYER_DIM.x, y = 58), 
            P3: Point(x = self.SCORE_PLAYER_DIM.x, y = 51), 
            P4: Point(x = self.SCORE_PLAYER_DIM.x, y = 58), 
            P5: Point(x = self.SCORE_PLAYER_DIM.x, y = 58)}[num_players]
        # distances between player-score text rectangles (HORIZONTAL, VERTICAL)
        self.SCORE_DIST = {
            P2: Point(7, 7), 
            P3: Point(6, 6), 
            P4: Point(7, 7), 
            P5: Point(7, 7)}[num_players]
        
        # Big white score background rectangle dim
        self.SCORE_BIG_DIM = Point(x= self.SCORE_PLAYER_DIM.x * num_players + self.SCORE_DIST.x * (num_players - 1), 
                              y=self.SCORE_PLAYER_DIM.y + self.SCORE_NUMBER_DIM.y + self.SCORE_DIST.y)
        # center point
        self.SCORE_BIG_POS = Point(WIDTH//2, HEIGHT//2 + 170) 
            
        # Player and score font
        self.SCORE_FONT_SIZE = 40


        # default winner text font size
        self.WIN_FONT_SIZE = 80 #  w1, w2, w3 - 10, w4 - 20 and w5 - 20
        # Winner text pos
        self.WIN_POS = Point(WIDTH//2, HEIGHT//2)

        # Winner screen logo scale
        self.WIN_LOGO_SCALE = 0.5