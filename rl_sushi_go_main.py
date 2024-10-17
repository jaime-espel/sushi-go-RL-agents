import numpy as np
import random
import os
import re
import sys

import gymnasium as gym
from gymnasium import spaces

from stable_baselines3 import A2C, DQN, PPO
from sb3_contrib import QRDQN, TRPO

from stable_baselines3.common.vec_env import DummyVecEnv # environment wrapper
from stable_baselines3.common.env_checker import check_env # Check environment
from stable_baselines3.common.callbacks import BaseCallback # Callbacks

import pygame
from settings import *
from utils import *
from Point import *
from animation import *

algorithm_dict = {
    'A2C': A2C, 
    'DQN': DQN, 
    'PPO': PPO, 
    'QR-DQN': QRDQN, 
    'TRPO': TRPO, 
}

agent_modes = ['RLAgent', 'RuleAgent', 'random', 'human', 'train'] # 'train' always at last position

class Card():
    def __init__(self, path, card_name):
        path_name = os.path.join(path, card_name)
        self.image = pygame.image.load(f'{path_name}.jpg').convert()

class Player:
    def __init__(self, id, player_mode = 'train'):
        self.id = id # int
        self.hand = np.zeros(12, dtype=int) # array_int[12]
        self.played_cards = np.zeros(15, dtype=int) # array_int[15]
        self.player_mode = player_mode # mode = ['RLAgent', 'random', 'human', 'train']
        self.player_model = None # Model of the player (for testing)
        
    # Basic puntuation (No-Makis, No-Pudin)
    def basic_score(self):
        score = 0
        # Nigiris
        score += self.played_cards[NIGIRI_TORTILLA]
        score += 2 * self.played_cards[NIGIRI_SALMON]
        score += 3 * self.played_cards[NIGIRI_CALAMAR]
        # Tempura y Sashimi
        score += (self.played_cards[TEMPURA] // 2) * 5
        score += (self.played_cards[SASHIMI] // 3) * 10
        # Gyoza
        if self.played_cards[GYOZA]: 
            score += [1, 3, 6, 10, 15][min(4, self.played_cards[GYOZA] - 1)]
        # Wasabi + Nigiri
        score += 3 * self.played_cards[NIGIRI_TORTILLA_WASABI]
        score += 6 * self.played_cards[NIGIRI_SALMON_WASABI]
        score += 9 * self.played_cards[NIGIRI_CALAMAR_WASABI]
        return score
    
    # Pick a valid card
    def pick_card(self, card):
        self.hand[card] -= 1 # Remove from hand
        # Add Card to played_cards
        if self.played_cards[WASABI] and card in (NIGIRI_TORTILLA, NIGIRI_SALMON, NIGIRI_CALAMAR):
            self.played_cards[WASABI] -= 1
            self.played_cards[wasabi_versions[card]] += 1
        else:
            self.played_cards[card] += 1
    
    # Choose a card via terminal
    def card_choice_human(self):
        print(f"\n Player{self.id}'s hand: ")
        for index, card in enumerate(self.hand):
            if card:
                print(f"\t{index}: {CARDS[index]} ({card})")
        while True:
            try:
                choice = int(input(f" Player{self.id}, pick a card of hand (0-{len(self.hand) - 1}): "))
                if 0 <= choice < len(self.hand) and self.hand[choice]: 
                    return choice
                else:
                    print(f"Invalid choice. Pick a number between 0 and {len(self.hand) - 1}.")
            except ValueError:
                print("Invalid input. Please enter a number.")
    

class SushiGoEnv(gym.Env):
    def __init__(self, ModelClass = PPO, num_players = 2):
        self.num_players = num_players                         # Number of players
        self.cards_per_player = 12 - num_players               # Cards per player
        self.total_rounds = 3                                  # Max rounds
        self.players = [Player(i) for i in range(num_players)] # List of players
        # For extensions
        self.diff_num_cards = 12    # 12 different cards
        self.played_cards_size = 15 # 15 possible plays
        self.max_num_players = 5

        # Action space: choose a card to play
        self.action_space = spaces.Discrete(self.diff_num_cards)

        # Observation space: hand + played cards
        self.total_size = (self.diff_num_cards + self.played_cards_size) * self.max_num_players
        self.observation_space = spaces.Box(low=-1, high=10, shape=(self.total_size,), dtype=int)

        # Previous model
        self.ModelClass = ModelClass
        self.model = None 
        
        # Observation of all the players
        self.obs_list = np.zeros(shape=(num_players, self.total_size), dtype=int)

        # Terminal or Pygame interface
        self.render_mode = None
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Turn
        self.pudin_in_turn= np.zeros(self.num_players, dtype=int)
        # Rounds
        self.rounds_played = 0
        self.pudin_in_round = np.zeros(self.num_players, dtype=int)
        # Deck and deal hands
        self.deck = DECK[:]
        random.shuffle(self.deck)
        self.deal_hands(INITIAL)
        # Scores
        self.scores = np.zeros(self.num_players, dtype=int)
        # State of step and observations
        self.state_chopsticks = False
        self.obs_list.fill(0)
        # Render initialization
        if self.render_mode == 'pygame':
            self.reset_pygame()
        return (self._get_obs(self.players[0]), {})
    
    # Dealing hands
    def deal_hands(self, deal_mode):
        for player in self.players:
            player.hand.fill(0)
            for _ in range(self.cards_per_player):
                player.hand[self.deck.pop()] += 1 
            if deal_mode == INITIAL: player.played_cards.fill(0)
            elif deal_mode == IN_GAME: player.played_cards[1:].fill(0)

    # Return observation for the player given
    def _get_obs(self, player):
        # Rotate players
        player_index = self.players.index(player)
        rotated_players = self.players[player_index:] + self.players[:player_index]
        
        cards_played = self.cards_per_player - int(np.sum(player.hand))
        min_hand_i = max(0, self.num_players - cards_played) % self.num_players
        # Initialize an empty list to collect observations
        observation_parts = []
        # For-each player
        for i, p in enumerate(rotated_players):
            # Add the hand 
            if i == 0 or (cards_played and  min_hand_i <= i):
                observation_parts.append(p.hand.copy())
            else:
                observation_parts.append(np.full(self.diff_num_cards, -1))
            # get their played cards
            observation_parts.append(p.played_cards.copy())
        for _ in range(self.max_num_players - self.num_players):
            # observation_parts.append(np.zeros(self.diff_num_cards + self.played_cards_size))
            observation_parts.append(np.full(self.diff_num_cards + self.played_cards_size, -1))
        # Concatenate all parts into a single observation array
        observation = np.concatenate(observation_parts)
        return observation
    
    # Load previous models
    def load_model(self, prev_model_path):
        self.model = self.ModelClass.load(prev_model_path, env=DummyVecEnv([lambda: self]))        

    def step(self, action):
        player = self.players[0] # We are training Player0
        has_chopsticks = sum(player.hand) >= 2 and player.played_cards[PALILLOS]
        reward = 0
        if not self.state_chopsticks:
            self.pudin_in_turn = np.zeros(self.num_players, dtype=int)
            # Get observations for all players --> before anything is done
            for i, player_i in enumerate(self.players):
                self.obs_list[i] = self._get_obs(player_i)
            # Check if the action is valid
            if player.hand[action] <= 0: 
                # Penalty
                reward = -  2 / ((12 - self.num_players) * 3)
                # If not valid action: Select a random valid action
                valid_actions = [i for i in range(len(player.hand)) if player.hand[i] > 0]
                action = random.choice(valid_actions)
            # Pick the card
            player.pick_card(action)
            if action == PUDIN: self.pudin_in_turn[0] += 1
            if has_chopsticks:
                self.state_chopsticks = True
                obs = self._get_obs(player)
                return obs, reward, False, False, {}
        else:  
            # state_chopstick
            self.state_chopsticks = False
            # if valid action: take it
            # else: do nothing (no negative reward)
            if player.hand[action]: 
                player.pick_card(action)
                player.hand[PALILLOS] += 1
                player.played_cards[PALILLOS] -= 1
                if action == PUDIN: self.pudin_in_turn[0] += 1

        # Opponent turns 
        for i, opponent_i in enumerate(self.players[1:], start=1): 
            # Pick a card
            opp_action = self.predict_opp_action(i)
            if opp_action == PUDIN: self.pudin_in_turn[i] += 1
            opponent_i.pick_card(opp_action)
            # Play Chopsticks
            if opp_action != PALILLOS or opponent_i.played_cards[PALILLOS] > 1:
                # Modify observations for Sushi Go!
                self.obs_list[i, :12] = self.players[i].hand.copy()
                pos_player_in_obs = 12
                self.obs_list[i, pos_player_in_obs: pos_player_in_obs + self.played_cards_size] = self.players[i].played_cards.copy()
                # Sushi Go!
                self.ShusiGo(i)
                
        # Update pudins
        self.pudin_in_round += self.pudin_in_turn
        self.pudin_in_turn = np.zeros(self.num_players, dtype=int)
        
        # End of the round (if hand is empty)
        done = False
        end_of_round = np.sum(player.hand) == 0  
        if end_of_round:
            self.scores += self.calculate_scores()
            self.rounds_played += 1
            if self.rounds_played < self.total_rounds:
                self.deal_hands(IN_GAME)
                self.pudin_in_round.fill(0)
            else:
                self.scores += self.calculate_pudin_scores()
                # Win or loose
                winner_indices = self.winner_indices()
                if 0 in winner_indices: # index_of_winner == 0 (we are training P0)
                    if len(winner_indices) == 1: reward += 1
                else:
                    reward -= 1
                done = True
        else:
            self.rotate_hands()

        # Obtain the new environment
        obs = self._get_obs(player)
        return obs, reward, done, False, {}

    def winner_indices(self):
        # max_score
        max_score = np.max(self.scores)
        # indices of the players with max_score
        max_indices = np.where(self.scores == max_score)[0] 
        # if one winner return
        if len(max_indices) == 1: return max_indices
        # played pudins of the winners
        winner_pudins = np.array([self.players[i].played_cards[PUDIN] for i in max_indices])
        # max_number of pudins in the winners
        max_pudins = np.max(winner_pudins)
        max_pudins_in_winners = np.where(winner_pudins == max_pudins)[0] 
        return max_indices[max_pudins_in_winners]
    
    def get_ranking(self):
        # Input Arrays
        scores = np.array(self.scores)
        pudin = np.array([self.players[i].played_cards[PUDIN] for i in range(self.num_players)])

        # Get the initial rankings based on score, descending order
        sorted_indices = np.argsort(-scores)
        rankings = np.empty_like(sorted_indices)
        # rankings[i] = j; --> P_i is at rank j
        rankings[sorted_indices] = np.arange(1, len(scores) + 1)

        # Handle ties based on scores and pudins
        unique_scores, counts = np.unique(scores, return_counts=True)
        for score, count in zip(unique_scores, counts):
            if count > 1:
                # Find players with the tied score
                tied_indices = np.where(scores == score)[0]
                tied_pudins = pudin[tied_indices]
                sorted_pudin_indices = tied_indices[np.argsort(-tied_pudins)]

                best_rank = min(rankings[sorted_pudin_indices])
                for i in range(len(tied_indices)):
                    if i == 0 or pudin[sorted_pudin_indices[i]] !=  pudin[sorted_pudin_indices[i - 1]]:
                        rankings[sorted_pudin_indices[i]] = best_rank + i # Assign rank based on the first tied player
                    else: 
                        rankings[sorted_pudin_indices[i]] = rankings[sorted_pudin_indices[i-1]]  # Assign same rank as previous if pudins are equal
        return rankings
    
    def predict_opp_action(self, i):
        opponent_i = self.players[i]
        match self.players[i].player_mode: # player_modes = ['RLAgent', 'random', 'human', 'train']
            case 'random':
                opp_valid_actions = [i for i in range(len(opponent_i.hand)) if opponent_i.hand[i] > 0]
                action_opp = random.choice(opp_valid_actions)
                return action_opp

            case 'RLAgent':
                if opponent_i.player_model:
                    obs_opp = self.obs_list[i]
                    action_opp, _ = opponent_i.player_model.predict(obs_opp, deterministic=True)
                    action_opp = int(action_opp)
                    if opponent_i.hand[action_opp]: 
                        return action_opp
                opp_valid_actions = [i for i in range(len(opponent_i.hand)) if opponent_i.hand[i] > 0]
                action_opp = random.choice(opp_valid_actions)
                return action_opp
            
            case 'RuleAgent':
                obs_opp = self.obs_list[i]
                action_opp = rule_action_selection(self, obs_opp)
                return action_opp

            case 'human':
                obs_opp = self.obs_list[i]
                if self.render_mode == 'print': action_opp = opponent_i.card_choice_human()
                elif self.render_mode == 'pygame': action_opp = self.render_human(i, obs_opp, False)
                return action_opp

            case 'train' | _:
                if self.model:
                    obs_opp = self.obs_list[i]
                    action_opp, _ = self.model.predict(obs_opp, deterministic=True)
                    action_opp = int(action_opp)
                    if opponent_i.hand[action_opp]: 
                        return action_opp

                opp_valid_actions = [i for i in range(len(opponent_i.hand)) if opponent_i.hand[i] > 0]
                action_opp = random.choice(opp_valid_actions)
                return action_opp
    
    def ShusiGo(self, i):
        player = self.players[i]
        if sum(player.hand) >= 1 and player.played_cards[PALILLOS]:
            
            match player.player_mode:
                case 'random':
                    action = random.randint(0, 11)
                    
                case 'RLAgent':
                    if not player.player_model: return
                    obs = self.obs_list[i]
                    action, _ = player.player_model.predict(obs, deterministic=True)
                    action = int(action)

                case 'RuleAgent':
                    obs = self.obs_list[i]
                    action = rule_action_selection(self, obs)
                        
                case 'human':
                    if self.render_mode == 'print': 
                        if input("Shusi Go? (y/n)").strip().lower() != 'y': return
                        action = player.card_choice_human()
                        
                    elif self.render_mode == 'pygame':
                        obs_i = self.obs_list[i]
                        action = self.render_human(i, obs_i, True)
                        if action == None: return

                case 'train' | _:
                    if not self.model: return
                    obs = self.obs_list[i]
                    action, _ = self.model.predict(obs, deterministic=True)
                    action = int(action)

            if player.hand[action]: 
                player.pick_card(action)
                player.hand[PALILLOS] += 1
                player.played_cards[PALILLOS] -= 1
                if action == PUDIN: self.pudin_in_round[i] += 1
    
    def rotate_hands(self):
        # Rotate hands (simulate passing hands to the left)
        temp_hand = self.players[0].hand
        for i in range(len(self.players) - 1):
            self.players[i].hand = self.players[i + 1].hand
        self.players[-1].hand = temp_hand

    def calculate_makis_scores(self):
        # Array of makis scores
        makis_scores = np.zeros(self.num_players, dtype=int) 

        # Array with number of makis per player
        makis = [player.played_cards[MAKI_1] + 2 * player.played_cards[MAKI_2] + 3 * player.played_cards[MAKI_3] for player in self.players]
        # Array of (Number of makis, player_index)
        makis_indices = [(makis[i], i) for i in range(self.num_players)]
        makis_indices.sort(key=lambda x: x[0], reverse=True) # Sort by the number of makis, descending

        # Max reward
        max_makis = makis_indices[0][0] # Determine max_makis
        max_makis_count = makis.count(max_makis) # Determine how many have max_makis
        first_place_points = 6 // max_makis_count
        for i in range(max_makis_count):
            makis_scores[makis_indices[i][1]] = first_place_points
        if max_makis_count > 1: return makis_scores # If more than one has max_makis, no second places
        makis_indices.pop(0) # Else: there is only one max (we delete it)
        if len(makis_indices) < 2: return makis_scores # if 2 players: just count max

        # Second max reward
        second_max_makis  = makis_indices[0][0] # Determine second max
        second_max_makis_count = makis.count(second_max_makis) # Determine how many have second max
        second_place_points = 3 // second_max_makis_count # Second_max reward
        for i in range(second_max_makis_count):
            makis_scores[makis_indices[i][1]] = second_place_points
        return makis_scores
    
    def calculate_scores(self):
        scores = np.array([player.basic_score() for player in self.players])
        return self.calculate_makis_scores() + scores
    
    def calculate_pudin_scores(self):
        pudin_scores = np.zeros(self.num_players, dtype=int) 
        pudin_indices = [(player.played_cards[PUDIN], player_idx) for player_idx, player in enumerate(self.players)]
        pudin_indices.sort(key=lambda x: x[0], reverse=True)

        # Max reward
        max_pudin = pudin_indices[0][0] # Determine max_pudin
        max_pudin_count = sum(1 for x in pudin_indices if x[0] == max_pudin) # Determine how many have max_pudin
        first_place_points = 6 // max_pudin_count
        for i in range(max_pudin_count):
            pudin_scores[pudin_indices[i][1]] = first_place_points

        # 2 players no penalization
        if self.num_players < 3: return pudin_scores 
        
        # Min reward
        min_pudin = pudin_indices[-1][0] # Determine min_pudin
        min_pudin_count = sum(1 for x in pudin_indices if x[0] == min_pudin) # Determine how many have max_pudin
        last_place_points = 6 // min_pudin_count
        for i in range(1, min_pudin_count + 1):
            pudin_scores[pudin_indices[-i][1]] -= last_place_points

        return pudin_scores

    def print(self):
        if self.rounds_played < 3: print( f'Round: {self.rounds_played}\n')
        for i, player in enumerate(self.players):
            print(f"Player{i}'s hand: {player.hand}")
            print(f"Player{i}'s played cards: {player.played_cards}")
        print()

    def init_pygame(self):
        # Configuration based on num_players
        self.Config = AppConfig(self.num_players)
        # Pygame initialization
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption(TITLE_STRING)
        # Clock to manage FPS
        self.clock = pygame.time.Clock()
        # Icon 
        icon_image = pygame.image.load('img/logo_m.png').convert_alpha()
        pygame.display.set_icon(icon_image)
        # Background
        self.background_surface = pygame.image.load('img/back_ground.jpg').convert()
        self.background_rect = self.background_surface.get_rect(midtop=(WIDTH//2, 0))
        # Scale the logo
        if self.Config.logo_scale:
            self.logo_img = pygame.image.load('img/logo.png').convert_alpha()
            self.logo_img = pygame.transform.rotozoom(self.logo_img, 0, self.Config.logo_scale)
            self.logo_rect = self.logo_img.get_rect(midtop=(WIDTH//2, 0))
        # Initialize hands
        self.cards = [Card(self.Config.card_path, CARDS[i]) for i in range(12)]
        self.hand_i = 0 # hand to render
        self.rects = [] # rects of hands' card

        # Hand/Player text font
        self.text_font = pygame.font.Font(self.Config.FONT_PATH, self.Config.FONT_SIZE)
        # Pudin text font
        self.pudin_font = pygame.font.Font(self.Config.FONT_PATH, self.Config.PUDIN_FONT_SIZE)
        # Round text font
        self.round_font = pygame.font.Font(self.Config.FONT_PATH, self.Config.ROUND_FONT_SIZE)

        # Create a semi-transparent surface (for player text rectangle)
        self.transparent_surface = pygame.Surface((self.Config.TEXT_W, self.Config.TEXT_H))
        self.transparent_surface.set_alpha(128)  # alpha: 0 (transparente) - 255 (opaco)

        # Transparent surface (size of (screen.width x hand_cards.bottom) )  # Optimized update of screen
        self.hand_layer = pygame.Surface((WIDTH, self.Config.POS_INI.y + self.Config.CARD_DIM.y), pygame.SRCALPHA)
        self.hand_layer.fill((0, 0, 0, 0))

        self.arrow_buttons()
        # For accumulation of pudins
        self.little_pudin_img = pygame.image.load('img/cards/little_pudin.png').convert_alpha()
        self.little_pudin_img = pygame.transform.rotozoom(self.little_pudin_img, 0, self.Config.l_pudin_scale)
        # Animation images
        self.your_turn_img_P = [pygame.image.load(f'img/your_turn/P{i}.png').convert_alpha() for i in range(self.num_players)]
        self.sushigo_img_P = [pygame.image.load(f'img/sushigo/P{i}.png').convert_alpha() for i in range(self.num_players)]
        # Initialize animations
        self.animations = {
            "turn_animation" : ScaleAnimation(0.3, 0.6, 0.008, 'center', Point(WIDTH // 2, HEIGHT // 2), self.screen),
            "sushigo_animation" : ScaleAnimation(0.3, 1.0, 0.015, 'topright_ref', self.Config.SUSHIGO_MESSAGE_POS, self.screen)
        }

        # "Listo" button
        self.listo_verde_img = pygame.image.load('img/listo/listo_verde.png').convert_alpha()
        self.listo_verde_img = pygame.transform.rotozoom(self.listo_verde_img, 0, self.Config.LISTO_SCALE)
        self.listo_amarillo_img = pygame.image.load('img/listo/listo_amarillo.png').convert_alpha()
        self.listo_amarillo_img = pygame.transform.rotozoom(self.listo_amarillo_img, 0, self.Config.LISTO_SCALE)
        self.listo_gris_img = pygame.image.load('img/listo/listo_gris.png').convert_alpha()
        self.listo_gris_img = pygame.transform.rotozoom(self.listo_gris_img, 0, self.Config.LISTO_SCALE)
        self.button_listo_rect = self.listo_verde_img.get_rect(center=self.Config.LISTO_POS.get_point()) 

        # Winner/Score rendering
        # Player and score number font
        self.score_font = pygame.font.Font(self.Config.FONT_PATH, self.Config.SCORE_FONT_SIZE)
        # Big white score background rectangle
        self.score_big_surf = pygame.Surface(self.Config.SCORE_BIG_DIM.get_point())
        self.score_big_surf.fill(WHITE)
        self.score_big_rect = self.score_big_surf.get_rect(center=self.Config.SCORE_BIG_POS.get_point())
        # Winner message font size depending of the number of winners: i winners --> win_font[i-1]
        self.win_font = [pygame.font.Font(self.Config.FONT_PATH, self.Config.WIN_FONT_SIZE),
                         pygame.font.Font(self.Config.FONT_PATH, self.Config.WIN_FONT_SIZE),
                         pygame.font.Font(self.Config.FONT_PATH, self.Config.WIN_FONT_SIZE - 10),
                         pygame.font.Font(self.Config.FONT_PATH, self.Config.WIN_FONT_SIZE - 20),
                         pygame.font.Font(self.Config.FONT_PATH, self.Config.WIN_FONT_SIZE - 20)] 
        # Logo
        if self.Config.WIN_LOGO_SCALE:
            self.win_logo_img = pygame.image.load('img/logo.png').convert_alpha()
            self.win_logo_img = pygame.transform.rotozoom(self.win_logo_img, 0, self.Config.WIN_LOGO_SCALE)
            self.win_logo_rect = self.win_logo_img.get_rect(midtop=(WIDTH//2, 0))
        # Draw puding margins (respect to the bottom-left corner of the score rectangle)
        self.score_pudin_img_margins = Point(48, 5) 
        self.score_pudin_txt_margin = Point(25, 4)
    
    
    def reset_pygame(self):
        self.hand_i = 0
        self.rects = [] # rects of hands' card
        self.listo_color = LISTO_GRIS
    
    def text_round(self):
        if self.rounds_played > 2: return
        # Top right round message
        text = f"Round: {self.rounds_played + 1}" 
        text_round_surf = self.round_font.render(text, True, WHITE)
        text_round_rect = text_round_surf.get_rect(topright = (WIDTH - self.Config.ROUND_X, self.Config.ROUND_Y))
        # Black text-background
        BG_MARGIN = self.Config.ROUND_BG_MARGIN
        l = text_round_rect.left - BG_MARGIN
        t = text_round_rect.top - BG_MARGIN
        w = text_round_rect.width + 2 * BG_MARGIN
        h = text_round_rect.height + 2 * BG_MARGIN
        black_bg_rect = pygame.Rect(l, t, w, h)
        # Render
        pygame.draw.rect(self.screen, BLACK, black_bg_rect)
        self.screen.blit(text_round_surf, text_round_rect)

    def text_rectangle(self, pos, player_i, screen, is_hand = False, is_truncated = False):
        if is_hand: 
            num_cards = 12 - self.num_players
            text = f'P{self.hand_i}.H'
        else: 
            num_cards = int(np.sum(self.players[player_i].played_cards[1:])) + self.pudin_in_round[player_i]
            if is_truncated and bool(num_cards): num_cards -= 1 # human players cant see what others have played before
            text = f'P{player_i}.P'

        # Big semi-transparent rectangle
        big_width = self.Config.TEXT_W + self.Config.CARD_DIM.x * num_cards + self.Config.CARD_DIST.x * num_cards
        BIG_transparent_surface = pygame.Surface((big_width, self.Config.TEXT_H))
        BIG_transparent_surface.set_alpha(128)  # Valores de alpha: 0 (transparente) a 255 (opaco)
        BIG_transparent_surface.fill(COLORS[player_i])
        
        # Text semi-transparent rectangle
        text_center_point = Point(x= pos[0] + self.Config.TEXT_W//2, y= pos[1] + self.Config.TEXT_H//2)
        text_surface = self.text_font.render(text, True, WHITE)
        text_rect = text_surface.get_rect(center = text_center_point.get_point())
        self.transparent_surface.fill(COLORS[player_i])
        
        # Render
        screen.blit(BIG_transparent_surface, pos)
        screen.blit(self.transparent_surface, pos)
        screen.blit(text_surface, text_rect)

        # Color lines for text rectangle
        rect = pygame.Rect(pos[0], pos[1], self.transparent_surface.get_width(), self.transparent_surface.get_height())  
        pygame.draw.rect(self.screen, COLORS[player_i], rect, width= 2) 

        # little pudins for played_cards
        if not is_hand:
            little_pudin_rect = self.little_pudin_img.get_rect(center=(text_center_point.x - self.Config.dist_text_puddin//2, text_center_point.y + self.Config.l_pudin_h))
            self.screen.blit(self.little_pudin_img, little_pudin_rect)
            if is_truncated:
                text = f"x{self.players[player_i].played_cards[0] - self.pudin_in_round[player_i] - self.pudin_in_turn[player_i]}" 
            else:
                text = f"x{self.players[player_i].played_cards[0] - self.pudin_in_round[player_i]}" 
            pudin_number_surf = self.pudin_font.render(text, True, BLACK)
            pudin_number_rect = pudin_number_surf.get_rect(midbottom = (text_center_point.x + self.Config.dist_text_puddin//2, little_pudin_rect.midbottom[1] + self.Config.diff_text_puddin))
            self.screen.blit(pudin_number_surf, pudin_number_rect)

    def played_text_render(self, trunc_rect_i = 0):
        Config = self.Config
        pos_act = Point(Config.POS_TEXT_INI.x - Config.desp, Config.POS_TEXT_INI.y + Config.TEXT_H + Config.CARD_DIST.y)
        for i in range(self.num_players):
            self.text_rectangle(pos_act.get_point(), i, self.screen, is_truncated= i < trunc_rect_i)
            pos_act.incr_y(Config.TEXT_H + Config.CARD_DIST.y)
    
    def hand_render(self):
        Config = self.Config
        # Hand text
        self.text_rectangle(Config.POS_TEXT_INI.get_point(), self.hand_i, self.hand_layer, True)
        # Hand graphics
        POS_ACT = Point(point=Config.POS_INI)
        player_h = self.players[self.hand_i]
        self.rects = []
        for i in range(len(player_h.hand)):
            for _ in range(player_h.hand[i]):
                rect = self.cards[i].image.get_rect(topleft = POS_ACT.get_point())
                self.rects.append((rect, i))
                self.hand_layer.blit(self.cards[i].image, rect)
                pygame.draw.rect(self.hand_layer, COLORS[self.hand_i], rect, width= 2) 
                POS_ACT.x += Config.CARD_DIM.x + Config.CARD_DIST.x
        # Print hand layer
        self.screen.blit(self.hand_layer, (0, 0))  
    
    # draw card borders
    def draw_card_rect(self, POS_ACT, player_i):
        rect = pygame.Rect(POS_ACT.x, POS_ACT.y, self.Config.CARD_DIM.x, self.Config.CARD_DIM.y)  
        pygame.draw.rect(self.screen, COLORS[player_i], rect, width= 2)

    def played_cards_render(self):
        Config = self.Config
        # Played cards lateral text
        self.played_text_render() 
        # Played cards graphics
        POS_ACT = Point(Config.POS_INI.x - self.Config.desp, Config.POS_INI.y + Config.CARD_DIM.y + Config.CARD_DIST.y)
        for player_i, player in enumerate(self.players):
            for i in range(len(player.played_cards)):
                if i == 0: # Pudin
                    for _ in range(self.pudin_in_round[player_i]):
                        self.screen.blit(self.cards[i].image, POS_ACT.get_point())
                        self.draw_card_rect(POS_ACT, player_i)
                        POS_ACT.x += Config.CARD_DIM.x + Config.CARD_DIST.x

                elif i < len(self.cards): # individual cards
                    for _ in range(player.played_cards[i]):
                        self.screen.blit(self.cards[i].image, POS_ACT.get_point())
                        self.draw_card_rect(POS_ACT, player_i)
                        POS_ACT.x += Config.CARD_DIM.x + Config.CARD_DIST.x
                else:
                    for _ in range(player.played_cards[i]): # Wasabi + Nigiri combinations
                        self.screen.blit(self.cards[WASABI].image, POS_ACT.get_point())
                        self.draw_card_rect(POS_ACT, player_i) 
                        self.screen.blit(self.cards[wasabi_inverse_versions[i]].image, (POS_ACT.x + 30, POS_ACT.y))
                        self.draw_card_rect(Point(POS_ACT.x + 30, POS_ACT.y), player_i) 
                        
                        POS_ACT.x += Config.CARD_DIM.x + Config.CARD_DIST.x

            POS_ACT.set_point(Config.POS_INI.x - self.Config.desp, POS_ACT.y + Config.CARD_DIM.y + Config.CARD_DIST.y)

    def arrow_buttons(self):
        # Load arrow images
        left_arrow_img = pygame.image.load('img/arrows/left-arrow.png')
        right_arrow_img = pygame.image.load('img/arrows/right-arrow.png')
        # Scale the arrow images
        left_arrow_img = pygame.transform.rotozoom(left_arrow_img, 0, self.Config.ARROW_SCALE)
        right_arrow_img = pygame.transform.rotozoom(right_arrow_img, 0, self.Config.ARROW_SCALE)
        # Get the rectangles in the center of the buttons
        left_img_rect = left_arrow_img.get_rect(center = (self.Config.BUTTON_DIM.x // 2, self.Config.BUTTON_DIM.y //2))
        right_img_rect = right_arrow_img.get_rect(center = (self.Config.BUTTON_DIM.x // 2, self.Config.BUTTON_DIM.y //2))
        # Create arrow buttons surfaces  
        self.l_arrow_button_surf = pygame.Surface(self.Config.BUTTON_DIM.get_point())
        self.r_arrow_button_surf = pygame.Surface(self.Config.BUTTON_DIM.get_point())
        # Color of the buttons
        self.l_arrow_button_surf.fill(GREY)
        self.r_arrow_button_surf.fill(GREY)
        # Copy the arrow images to the buttons
        self.l_arrow_button_surf.blit(left_arrow_img, left_img_rect)
        self.r_arrow_button_surf.blit(right_arrow_img, right_img_rect)
        # Position buttons
        self.l_arrow_button_rect = self.l_arrow_button_surf.get_rect(topright=self.Config.POS_TEXT_INI.get_point())
        num_cards = 12 - self.num_players
        big_width = self.Config.POS_TEXT_INI.x + self.Config.TEXT_W + self.Config.CARD_DIM.x * num_cards + self.Config.CARD_DIST.x * num_cards
        self.r_arrow_button_rect = self.r_arrow_button_surf.get_rect(topleft=(big_width, self.Config.POS_TEXT_INI.y))
        # Create arrow_rects array (to check collisions)
        self.arrow_rects = [self.l_arrow_button_rect, self.r_arrow_button_rect]
    
    # AGENTS RENDERING
    def render(self, render_mode=None):
        # render_mode = [None, "print", "pygame"]
        self.render_mode = render_mode
        if not render_mode: 
            return
        if render_mode == 'print':
            self.print()
            return
        if render_mode != 'pygame': return

        if not pygame.get_init():
            self.init_pygame()
        
        human_players_counter = sum([player.player_mode == "human" for player in self.players])
        if human_players_counter: return

        first_time = True
        has_changed = True
        can_apply_arrow = True

        while True:
            # Event loop
            for event in pygame.event.get(): 
                # press the X of the window
                if event.type == pygame.QUIT: 
                    pygame.quit()
                    sys.exit()

                # pressed keyboard
                if event.type == pygame.KEYDOWN: 
                    match event.key: 
                        # SPACE: exit 
                        case pygame.K_SPACE: 
                            return
                        # RIGHT ARROW: change viewing hand (next)
                        case pygame.K_RIGHT:
                            self.hand_i = (self.hand_i + 1) % self.num_players
                            has_changed = True
                        # LEFT ARROW: change viewing hand (previous)
                        case pygame.K_LEFT:
                            self.hand_i = (self.hand_i - 1 + self.num_players) % self.num_players
                            has_changed = True

                # Mouse click
                if event.type == pygame.MOUSEBUTTONDOWN:
                    pressed_buttons = pygame.mouse.get_pressed(num_buttons=3) # (is_Left_clicked, is_Middle_clicked, is_Right_clicked)
                    # Left button pressed
                    if pressed_buttons[0]: 
                        # Obtain mouse positon
                        mouse_pos = pygame.mouse.get_pos()
                        # Check collision for arrow buttons
                        if self.arrow_rects[0].collidepoint(mouse_pos) and can_apply_arrow:
                            self.hand_i = (self.hand_i - 1 + self.num_players) % self.num_players
                            has_changed = True
                            can_apply_arrow = False
                        if self.arrow_rects[1].collidepoint(mouse_pos) and can_apply_arrow:
                            self.hand_i = (self.hand_i + 1) % self.num_players
                            has_changed = True
                            can_apply_arrow = False

                # Mouse unclicked
                if event.type == pygame.MOUSEBUTTONUP:
                    pressed_buttons = pygame.mouse.get_pressed(num_buttons=3)
                    if not pressed_buttons[0]: 
                        can_apply_arrow = True
            
            # First while loop                  
            if first_time:
                # Background image
                self.screen.blit(self.background_surface, self.background_rect) 
                # Logo
                if self.Config.logo_scale:
                    self.screen.blit(self.logo_img, self.logo_rect) 
                # Hand arrows
                self.screen.blit(self.l_arrow_button_surf, self.l_arrow_button_rect)
                self.screen.blit(self.r_arrow_button_surf, self.r_arrow_button_rect)
                # Played cards
                self.played_cards_render()
                self.text_round()
                first_time = False
            
            # Every time the hand changes (also the first time)
            if has_changed:
                self.hand_render()
                self.hand_layer.fill((0, 0, 0, 0))
                has_changed = False
                pygame.display.update()

            self.clock.tick(FPS)

    # HUMAN RENDERING
    # Render observed played cards
    def obs_played_card_render(self, player_index, obs_of_index):
        Config = self.Config
        # Played cards left-lateral text (trucated big-transparent-rectangle)
        self.played_text_render(trunc_rect_i= player_index)
        # Played cards graphics
        POS_ACT = Point(Config.POS_INI.x - self.Config.desp, Config.POS_INI.y + Config.CARD_DIM.y + Config.CARD_DIST.y)
        # Positon of player 0
        pos_P0 = self.num_players - player_index
        # For each player render the corresponding played cards
        for player_i in range(self.num_players):
            # Calculate the actual positon of player_i's played cards inside the observation
            index = (player_i + pos_P0) % self.num_players
            ini = (self.diff_num_cards + self.played_cards_size) * index + self.diff_num_cards
            played_cards = obs_of_index[ini: ini + self.played_cards_size]

            # Do the rendering
            for card_i in range(len(played_cards)):
                if card_i == PUDIN: # card_i == 0
                    if player_i == player_index:
                        for _ in range(self.pudin_in_round[player_i]):
                            self.screen.blit(self.cards[card_i].image, POS_ACT.get_point())
                            self.draw_card_rect(POS_ACT, player_i)
                            POS_ACT.x += Config.CARD_DIM.x + Config.CARD_DIST.x
                    else:
                        for _ in range(self.pudin_in_round[player_i]):
                            self.screen.blit(self.cards[card_i].image, POS_ACT.get_point())
                            self.draw_card_rect(POS_ACT, player_i)
                            POS_ACT.x += Config.CARD_DIM.x + Config.CARD_DIST.x

                elif card_i < len(self.cards): # Rest of individual cards
                    for _ in range(played_cards[card_i]):
                        self.screen.blit(self.cards[card_i].image, POS_ACT.get_point())
                        self.draw_card_rect(POS_ACT, player_i)
                        POS_ACT.x += Config.CARD_DIM.x + Config.CARD_DIST.x
                else:
                    for _ in range(played_cards[card_i]): # Combinations of Wasabi + Nigiri
                        self.screen.blit(self.cards[WASABI].image, POS_ACT.get_point())
                        self.draw_card_rect(POS_ACT, player_i) 
                        self.screen.blit(self.cards[wasabi_inverse_versions[card_i]].image, (POS_ACT.x + 30, POS_ACT.y))
                        self.draw_card_rect(Point(POS_ACT.x + 30, POS_ACT.y), player_i)
                        POS_ACT.x += Config.CARD_DIM.x + Config.CARD_DIST.x

            POS_ACT.set_point(Config.POS_INI.x - self.Config.desp, POS_ACT.y + Config.CARD_DIM.y + Config.CARD_DIST.y)

    # "Your turn!" animation for multiple human players 
    def your_turn_animation(self, i):
        TURN_SCALE_ANIMATION = True
        SCALE_UP = True
        scale = 0.3
        while True:
            for event in pygame.event.get(): 
                if event.type == pygame.QUIT: # press the X of the window
                    pygame.quit()
                    sys.exit()
                 
                if event.type == pygame.KEYDOWN: # pressed keyboard
                    match event.key: 
                        case pygame.K_SPACE: # SPACE
                                return
            
            if TURN_SCALE_ANIMATION and SCALE_UP:
                self.screen.blit(self.background_surface, self.background_rect)
                P = pygame.transform.rotozoom(self.your_turn_img_P[i], 0, scale)
                self.screen.blit(P, P.get_rect(center=(WIDTH//2, HEIGHT//2)))
                scale += 0.008
                if scale > 1:
                    SCALE_UP = False
            elif TURN_SCALE_ANIMATION and not SCALE_UP:
                self.screen.blit(self.background_surface, self.background_rect)
                P = pygame.transform.rotozoom(self.your_turn_img_P[i], 0, scale)
                self.screen.blit(P, P.get_rect(center=(WIDTH//2, HEIGHT//2)))
                scale -= 0.008
                if scale < 0.3 :
                    TURN_SCALE_ANIMATION = False
                    return
                
            pygame.display.update()
            self.clock.tick(FPS)


    def render_human(self, i, obs_i, sushigo):
        # Current hand to render
        self.hand_i = i
        # First while loop (also True during animations)
        first_time = True
        # Hand selection update
        has_changed = True
        # 'Listo' button update
        has_changed_button = True

        # Selection flag
        can_select = True
        # Selected card (checking collisions)
        selected = None
        selected_rect = None
    
        # Control of 'pygame.display.update()'
        UPDATE_SCREEN = True
        # Active animation flag
        ANY_ACTIVE_ANIMATION = True

        # Animation rendering
        if not sushigo: # Normal pick of cards
            human_players_counter = sum([player.player_mode == "human" for player in self.players])
            if human_players_counter > 1:
                self.your_turn_animation(i)
            else: self.animations["turn_animation"].reset()
            self.listo_color = LISTO_GRIS
        else:  # Sushi Go!
            self.animations["sushigo_animation"].reset()
            self.listo_color = LISTO_AMARILLO
            
        while True:
            # Event loop
            for event in pygame.event.get(): 
                # press the X of the window
                if event.type == pygame.QUIT: 
                    pygame.quit()
                    sys.exit()

                # pressed keyboard
                if event.type == pygame.KEYDOWN: 
                    match event.key: 
                        # SPACE
                        case pygame.K_SPACE: 
                            if selected != None or sushigo: 
                                # Stop animations
                                for _, value in self.animations.items(): value.is_active = False
                                return selected
                            
                # Mouse click
                if event.type == pygame.MOUSEBUTTONDOWN:
                    pressed_buttons = pygame.mouse.get_pressed(num_buttons=3) # (is_Left_clicked, is_Middle_clicked, is_Right_clicked)
                    # Left button pressed
                    if pressed_buttons[0]: 
                        # Obtain mouse positon
                        mouse_pos = pygame.mouse.get_pos()
                        # Check (mouse-card) collision for-each card in the hand
                        for card_rect, card_i in self.rects:
                            if card_rect.collidepoint(mouse_pos) and can_select: 
                                # if the selected card is clicked: unselect it
                                # else: select the clicked card
                                if card_rect == selected_rect: 
                                    selected = None
                                    selected_rect = None
                                    if sushigo: self.listo_color = LISTO_AMARILLO
                                    else: self.listo_color = LISTO_GRIS
                                else:
                                    selected = card_i
                                    selected_rect = card_rect
                                    # print(f'Player{self.hand_i}; Selected card: {CARDS[card_i]}; Selected_rect: {selected_rect}')
                                    self.listo_color = LISTO_VERDE
                                has_changed = True
                                has_changed_button = True
                                can_select = False
                        # Check (mouse-"Listo" button) collision       
                        if self.button_listo_rect.collidepoint(mouse_pos) and can_select:
                            if selected != None or sushigo: # if is SushiGo! you can return None
                                return selected
                            can_select = False

                # Mouse unclicked           
                if event.type == pygame.MOUSEBUTTONUP:
                    pressed_buttons = pygame.mouse.get_pressed(num_buttons=3)
                    if not pressed_buttons[0]: 
                        can_select = True

            # Check for active animations
            ANY_ACTIVE_ANIMATION = sum([1 if value.is_active else 0 for _, value in self.animations.items()])
                                
            if first_time or ANY_ACTIVE_ANIMATION:
                # Background image
                self.screen.blit(self.background_surface, self.background_rect) 
                # Logo render
                if self.Config.logo_scale: self.screen.blit(self.logo_img, self.logo_rect) 
                # Played cards render
                self.obs_played_card_render(i, obs_i)
                # Round render
                self.text_round()
                first_time = False

            # Hand rendering
            if has_changed or ANY_ACTIVE_ANIMATION:
                self.hand_render()
                if selected_rect:
                    pygame.draw.rect(self.hand_layer, VERDE, selected_rect, 5)
                    self.screen.blit(self.hand_layer, (0, 0))
                self.hand_layer.fill((0, 0, 0, 0))
                has_changed = False
                UPDATE_SCREEN = True

            # "Listo" button rendering
            if has_changed_button or ANY_ACTIVE_ANIMATION:
                match self.listo_color:
                    case 0: self.screen.blit(self.listo_verde_img, self.button_listo_rect)
                    case 1: self.screen.blit(self.listo_amarillo_img, self.button_listo_rect)
                    case _: self.screen.blit(self.listo_gris_img, self.button_listo_rect)

                has_changed_button = False
                UPDATE_SCREEN = True
            
            # "Your turn!" animation
            if self.animations["turn_animation"].is_active:
                self.animations["turn_animation"].animate(self.your_turn_img_P[i])
                if not self.animations["turn_animation"].is_active:
                    first_time, has_changed, has_changed_button = True, True, True
                UPDATE_SCREEN = True

            # "Sushi Go!" animation
            if self.animations["sushigo_animation"].is_active and not self.animations["turn_animation"].is_active:
                self.animations["sushigo_animation"].animate(self.sushigo_img_P[i])
                if not self.animations["sushigo_animation"].is_active:
                    first_time, has_changed, has_changed_button = True, True, True
                UPDATE_SCREEN = True

            # Update changes
            if UPDATE_SCREEN:
                pygame.display.update()
                UPDATE_SCREEN = False
            
            self.clock.tick(FPS)
    
    def render_scores(self):
        if not self.render_mode: 
            return
        if self.render_mode == 'print':
            # Winner text
            winner_list = self.winner_indices()
            if len(winner_list) == 1: 
                text = f"Player{winner_list[0]} Wins!" 
            else:
                winners_text = ", ".join(f"P{winner_i}" for winner_i in winner_list[:-1])
                text = f"Winners: {winners_text} and P{winner_list[-1]}!"
            print(text)
            return
        if self.render_mode != 'pygame': return

        """
            Player0 Wins! 
        ---------------------
        | P0 | P1 | P2 | P3 |
        ---------------------
        | 20 |  2 | 40 | 50 |
        ---------------------
        """
        Config = self.Config
        first_time = True

        while True:
            # Event loop
            for event in pygame.event.get(): 
                # press the X of the window: Exit application
                if event.type == pygame.QUIT: 
                    pygame.quit()
                    sys.exit()
                # pressed keyboard
                if event.type == pygame.KEYDOWN: 
                    match event.key:
                        # SPACE: Exit
                        case pygame.K_SPACE: return 
                # Mouse click
                if event.type == pygame.MOUSEBUTTONDOWN:
                    pressed_buttons = pygame.mouse.get_pressed(num_buttons=3) # (is_Left_clicked, is_Middle_clicked, is_Right_clicked)
                    # Left button pressed
                    if pressed_buttons[0]: return

            # First while loop                              
            if first_time:
                # Background image
                self.screen.blit(self.background_surface, self.background_rect) 
                # Score white background rect
                self.screen.blit(self.score_big_surf, self.score_big_rect)
                # Winner text
                winner_list = self.winner_indices()
                if len(winner_list) == 1: 
                    text = f"Player{winner_list[0]} Wins!" 
                else:
                    winners_text = ", ".join(f"P{winner_i}" for winner_i in winner_list[:-1])
                    text = f"Winners: {winners_text} and P{winner_list[-1]}!"
                # Choose fontsize depending on the number of winners
                win_text_surf = self.win_font[len(winner_list) - 1].render(text, True, WHITE)
                win_text_rect = win_text_surf.get_rect(center=self.Config.WIN_POS.get_point())
                
                # Crea una superficie medio transparente negra para el background del texto
                transparent_surface = pygame.Surface((win_text_rect.w + 20, win_text_rect.h + 20))
                transparent_surface.set_alpha(175)  # alpha: 0 (transparente) - 255 (opaco)
                transparent_surface.fill(BLACK)
                transparent_rect = transparent_surface.get_rect(center=self.Config.WIN_POS.get_point())
                # Background rectangle to screen
                self.screen.blit(transparent_surface, transparent_rect) 
                # Text to screen
                self.screen.blit(win_text_surf, win_text_rect)

                # Fill the table with: Titles (P_i) and Scores (int: at most 3 digits)
                for i in range(self.num_players):
                    # Player surface
                    score_player_surf = pygame.Surface(Config.SCORE_PLAYER_DIM.get_point())
                    score_player_surf.fill(COLORS[i])
                    score_player_rect = score_player_surf.get_rect(topleft=(self.score_big_rect.left + (Config.SCORE_PLAYER_DIM.x + Config.SCORE_DIST.x) * i, self.score_big_rect.top))
                    # Player text
                    player_text_surf = self.score_font.render(f'P{i}', True, WHITE)
                    player_text_rect = player_text_surf.get_rect(center=(score_player_rect.w//2, score_player_rect.h//2))
                    score_player_surf.blit(player_text_surf, player_text_rect)
                    # Player surface + Player text
                    self.screen.blit(score_player_surf, score_player_rect)
                    # Number surface
                    score_number_surf = pygame.Surface(Config.SCORE_NUMBER_DIM.get_point())
                    score_number_surf.fill(BLACK)
                    score_number_rect = score_number_surf.get_rect(topleft=(self.score_big_rect.left + (Config.SCORE_PLAYER_DIM.x + Config.SCORE_DIST.x) * i, self.score_big_rect.top + (Config.SCORE_PLAYER_DIM.y + Config.SCORE_DIST.y)))
                    # Number text
                    number_text_surf = self.score_font.render(f'{self.scores[i]:3}' if self.scores[i] > 9 else f' {self.scores[i]} ', True, WHITE)
                    number_text_rect = number_text_surf.get_rect(center=(score_number_rect.w//2, score_number_rect.h//2))
                    score_number_surf.blit(number_text_surf, number_text_rect)
                    # Number surface + Number text
                    self.screen.blit(score_number_surf, score_number_rect)
                    # Little pudins in score (only if there is a tie in scores)
                    max_score = np.max(self.scores)
                    max_indices = np.where(self.scores == max_score)[0] 
                    if len(max_indices) > 1 and i in max_indices:
                        little_pudin_rect = self.little_pudin_img.get_rect(bottomleft=(score_number_rect.right - self.score_pudin_img_margins.x, 
                                                                                    score_number_rect.bottom - self.score_pudin_img_margins.y))
                        self.screen.blit(self.little_pudin_img, little_pudin_rect)
                        text = f'x{self.players[i].played_cards[PUDIN]:2}' if self.players[i].played_cards[PUDIN] > 9 else f' x{self.players[i].played_cards[PUDIN]} '
                        pudin_number_surf = self.pudin_font.render(text, True, WHITE)
                        pudin_number_rect = pudin_number_surf.get_rect(center=(little_pudin_rect.centerx + self.score_pudin_txt_margin.x, 
                                                                                    little_pudin_rect.centery + self.score_pudin_txt_margin.y))
                        self.screen.blit(pudin_number_surf, pudin_number_rect)
                # Big white border 
                big_border_rect = pygame.Rect(self.score_big_rect.left - self.Config.SCORE_DIST.x, 
                            self.score_big_rect.top - self.Config.SCORE_DIST.x,
                            self.score_big_rect.w + 2 * self.Config.SCORE_DIST.x,
                            self.score_big_rect.h + 2 * self.Config.SCORE_DIST.x)
                pygame.draw.rect(self.screen, WHITE, big_border_rect, self.Config.SCORE_DIST.x)
                # Logo
                if self.Config.WIN_LOGO_SCALE: self.screen.blit(self.win_logo_img, self.win_logo_rect) 
                
                first_time = False
                pygame.display.update()

            self.clock.tick(FPS)
        
        
    def close(self):
        # Close the rendering window.
        if pygame.get_init(): 
            pygame.quit()
      
# Customized Callback
class LastCheckpointCallback(BaseCallback):
    def __init__(self, env, save_freq_ep, max_episodes, save_path, name='rl_model', save_replay_buffer=False, save_vecnormalize=False, verbose=0, prev_episode_number=0, model_P_ep_save_freq = -1):
        super().__init__(verbose)
        
        self.env = env # environment
        self.save_freq_ep = save_freq_ep # training model save/load frequency (in episodes/games played)
        self.save_path = save_path # intermediate model_path
        self.name = name # name of the intermediate models
        self.save_replay_buffer = save_replay_buffer
        self.save_vecnormalize = save_vecnormalize
        
        self.episodes = 0 # Finished episodes
        self.max_episodes = max_episodes # Maximum number of episodes
        self.prev_episode_number = prev_episode_number
        
        # Resultant models
        self.model_P_ep_save_freq = model_P_ep_save_freq


    def _on_step(self) -> bool:
        episode_done = self.locals['dones'][0]
        result = super()._on_step()
        # Check if the episode is done (played a game)
        if episode_done:
            self.episodes += 1
            # Every save_freq_ep
            if self.save_freq_ep > 0 and self.episodes % self.save_freq_ep == 0:
                # print('\rEpisodes:', self.episodes, end="")
                model_path = os.path.join(self.save_path, f"{self.name}.zip")
                # Remove previous model
                if os.path.exists(model_path):
                    os.remove(model_path)
                # Save model
                if result and self.episodes < self.max_episodes: # Training not finished
                    # Use default name path
                    self.model.save(model_path)
                    # Load model as opponent
                    model_path = model_path[:-4]
                    self.env.load_model(model_path)
                    
                else: # Training  finished
                    # Use name with information, for example: 'PPO_model_P2_ep150000.zip'
                    episodes_in_name = self.episodes + self.prev_episode_number
                    model_path = os.path.join(self.save_path, f"{self.name}_P{self.env.num_players}_ep{episodes_in_name}.zip")

                    # If the name exists add (number)
                    if os.path.exists(model_path):
                        i = 1
                        model_path_name = os.path.join(self.save_path, f"{self.name}_P{self.env.num_players}_ep{episodes_in_name}({i})")
                        while os.path.exists(model_path_name + '.zip'):
                            i += 1
                            model_path_name = os.path.join(self.save_path, f"{self.name}_P{self.env.num_players}_ep{episodes_in_name}({i})")
                        model_path = model_path_name + '.zip'
                    self.model.save(model_path)
                

            if self.episodes < self.max_episodes and self.model_P_ep_save_freq != -1 and self.episodes % self.model_P_ep_save_freq == 0: # result model save frequency
                # Use name with information, for example: 'PPO_model_P2_ep150000.zip'
                episodes_in_name = self.episodes + self.prev_episode_number
                model_path = os.path.join(self.save_path, f"{self.name}_P{self.env.num_players}_ep{episodes_in_name}.zip")

                # If the name exists add (number)
                if os.path.exists(model_path):
                    i = 1
                    model_path_name = os.path.join(self.save_path, f"{self.name}_P{self.env.num_players}_ep{episodes_in_name}({i})")
                    while os.path.exists(model_path_name + '.zip'):
                        i += 1
                        model_path_name = os.path.join(self.save_path, f"{self.name}_P{self.env.num_players}_ep{episodes_in_name}({i})")
                    model_path = model_path_name + '.zip'
                self.model.save(model_path)

        return result and self.episodes < self.max_episodes


def choose_RL_algorithm():
    valid_algorithm = False
    while not valid_algorithm:
        print('Supported algorithms:')
        supported_algorithms = list(algorithm_dict.keys())
        for indice, alg in enumerate(supported_algorithms):
            print(f"\t{indice}: {alg} ")
        algorithm_number = input("Enter algorithm number:").strip()
        if algorithm_number.isnumeric() and (algorithm_number := int(algorithm_number)) in range(len(supported_algorithms)) :
            ModelClass = algorithm_dict[supported_algorithms[algorithm_number]]
            valid_algorithm = True
    algorithm_name = supported_algorithms[algorithm_number]
    print(f'\nUsing {algorithm_name}')
    return ModelClass, algorithm_name

# Training 
def train_agent(numPlayers):
    # Resulting model-path
    model_path = os.path.join('Training', 'Model')
    os.makedirs(model_path, exist_ok=True)

    # Choose algorithm
    ModelClass, algorithm_name = choose_RL_algorithm()
    name = algorithm_name + '_model'

    # Total number of episodes (games) to train
    total_episodes = int(input("Enter total episodes/games of training: "))
    # Frequency for saving past versions of the model as opponents
    save_freq = int(input("Enter save-frequency of the model as training opponents (in episodes/games): "))
    # Intermediate models saving
    has_intermediate = input("Do you want to save intermediate models? (y/n): ")
    if has_intermediate.lower() == 'y' or has_intermediate.lower() == 'yes':
        # Resultant models save frequency
        model_P_ep_save_freq = int(input("Enter save-frequency of intermediate models (in episodes/games): "))
    else:
        model_P_ep_save_freq = -1

    # Create environment 
    env = SushiGoEnv(ModelClass=ModelClass, num_players=numPlayers)
    # Create callback
    checkpoint_callback = LastCheckpointCallback(
        env = env, 
        save_freq_ep = save_freq, 
        max_episodes = total_episodes, 
        save_path = model_path, 
        name = name,
        model_P_ep_save_freq=model_P_ep_save_freq
        )
    # Calculate maximum number of timesteps
    max_timesteps = 60 * total_episodes # 10 cards * 3 rounds *  2 (sushigo option)

    # Wrap the environment
    env = DummyVecEnv([lambda: env]) 
    # Create Model
    model = ModelClass('MlpPolicy', env, verbose = 0)
 
    # Do the actual training
    model.learn(total_timesteps=max_timesteps , callback=checkpoint_callback)
    env.close()

"""
TEST_AGENT

for-each game:
    reset environment 
    while not ended:
        render
        choose model action
        step
        update reward
    render
    render_scores
 """  
def test_agent(env: SushiGoEnv, model, render_mode):
    # Obtain number of episodes/games to play
    episodes = input("Enter episode number, or games to play (q to quit): ").strip()
    if episodes == 'q': return False
    if episodes == '0': return True
    episodes = int(episodes) if episodes.isnumeric() else 1000 # antes a 1

    wins = 0
    for episode in range(1, episodes + 1): 
        obs, _ = env.reset() # --> (observation, info)
        ended = False
        score = 0
        while not ended:
            reward = 0
            # SushiGo!: render the last state
            if sum(env.players[0].hand) == sum(env.players[1].hand) and sum(env.players[0].played_cards[:12]) + 2 * sum(env.players[0].played_cards[12:]) == sum(env.players[1].played_cards[:12]) + 2 * sum(env.players[1].played_cards[12:]): 
                env.render(render_mode) 
            action, _ = model.predict(obs, deterministic=True) # --> (action, _states)
            obs, reward, done, truncated, info = env.step(action.item()) 
            score += reward
            ended = done or truncated 
        env.render(render_mode) 
        env.render_scores()

        winner_ind = env.winner_indices()
        if 0 in winner_ind and len(winner_ind) == 1: wins += 1

    print(f'Win rate {wins}/{episodes}: {wins/episodes:.2f}')
        
    env.close()
    return True

# Key for sorting model names
def sort_models_key(filename):
    # Extract the base name and episode number from the filename
    match = re.match(r"([-\w]+)_P(\d+)_ep(\d+)", filename)
    if match: # full name
        base_name = match.group(1)
        num_players = int(match.group(2))
        episode_number = int(match.group(3))
        # Check parentheses
        paren_match = re.search(r"\((\d+)\)", filename)
        if paren_match: paren_number = int(paren_match.group(1))
        else: paren_number = 0
        return (base_name, num_players, episode_number, paren_number)
    else: # just the name with episode numbers
        raise "Not valid model name"


# Obtain the model path for one of the available Models
def obtain_model_path():
    model_path = os.path.join('Training', 'Model')
    # Files in ./Training/Model/
    archivos = os.listdir(model_path)
    archivos_zip = [archivo for archivo in archivos if archivo.endswith('.zip')]
    # Sorted zips
    sorted_zips = sorted(archivos_zip, key=sort_models_key)
    # Print de Models
    print('\nAvailable Models: ')
    for indice, archivo in enumerate(sorted_zips):
        print(f"{indice}: {archivo[:-4]} ")
    model_choice = input("Enter your model choice: ").strip()
    while not (model_choice.isnumeric() and int(model_choice) in range(len(sorted_zips))):
        model_choice = input("Enter your model choice: ").strip()
    # Obtain model path
    model_choice = int(model_choice)
    model_name = sorted_zips[model_choice]
    print(f'\n\tUsing Model {model_name}\n')
    return os.path.join(model_path, model_name)

# Obtain a model in the folder "./Training/Model"
def obtain_model(env):
    # Obtain the path of the model
    model_path = obtain_model_path() 
    # Load the model, based on its name
    model_path_zip = os.path.basename(model_path)
    algorithm_name = model_path_zip[:model_path_zip.find('_')]
    if algorithm_name in algorithm_dict:
        ModelClass = algorithm_dict[algorithm_name]
    else: # default: PPO
        ModelClass = PPO
    return ModelClass.load(model_path, env=DummyVecEnv([lambda: env]))


# Obtain a model in the folder "./Training/Model"
def obtain_model_info():
    # Obtain the path of the model: 'Training\\Model\\QR-DQN_model_P2_ep1200000.zip'
    model_path_name = obtain_model_path() 
    # Load the model, based on its name: 'QR-DQN_model_P2_ep1200000.zip'
    model_path_basename = os.path.basename(model_path_name)

    match = re.match(r"([-\w]+)_P(\d+)[_\w]*_ep(\d+)", model_path_basename)
    if match: # full name
        base_name = match.group(1) # 'QR-DQN_model'
        index = base_name.index('_')
        algorithm_name = base_name[:index]
        ModelClass = algorithm_dict[algorithm_name]
        num_players = int(match.group(2)) # '2'
        prev_episode_number = int(match.group(3)) # '1200000'
        return model_path_name, ModelClass, algorithm_name, num_players, prev_episode_number
    else: raise ValueError("Incorrect name format: " + model_path_basename)




# Choose agent_mode: ['RLAgent', 'random', 'human', 'training']
def choose_opponent_mode(env: SushiGoEnv, i):
    print(f'Choose mode for P{i}:')
    for opt, agent_mode in enumerate(agent_modes[:-1]):
        print(f'\t{opt}. {agent_mode}')
    opt = input("Enter your choice: ").strip()
    while not (opt.isnumeric() and (opt := int(opt)) in range(len(agent_modes) - 1)):
        opt = input("Enter your choice: ").strip()
    # agent_modes = ['RLAgent', 'RuleAgent', 'random', 'human', 'train']
    env.players[i].player_mode = agent_modes[opt]
    if env.players[i].player_mode == 'RLAgent':
        env.players[i].player_model = obtain_model(env)
    print()

# Choose num_players: [2, 3, 4, 5]
def ask_for_num_players():
    while not (user_input := input("Enter the number of players: ")).isdigit() or not (2 <= (num_players := int(user_input)) <= 5) :
        print("Invalid input. Please enter a number [2, 5].")
    return num_players

# Choose render_mode: ['print', 'pygame']
def ask_render_mode():
    render_modes = ['print', 'pygame']
    print('Render modes:')

    for n, mode in enumerate(render_modes):
        print(f'\t{n}. {str(mode)}')

    while not (user_input := input("Choose render mode: ")).isdigit() or not (0 <= (render_mode := int(user_input)) <= 2) :
        for n, mode in enumerate(render_modes):
            print(f'\t{n}. {str(mode)}')
    return render_modes[render_mode]


CARDS_SCORE_RANKING = [
    # NIGIRI
    NIGIRI_CALAMAR,
    NIGIRI_SALMON,
    # TEMPURA
    TEMPURA,
    # MAKI
    MAKI_3,
    MAKI_2,
    # SETS
    GYOZA,
    SASHIMI,
    # REST
    PUDIN,
    NIGIRI_TORTILLA,
    MAKI_1,
    PALILLOS,
    WASABI,
]

def rule_action_selection(env: SushiGoEnv, obs):
    my_hand = obs[:env.diff_num_cards]
    my_played = obs[env.diff_num_cards : env.diff_num_cards + env.played_cards_size]
    len_hand = sum(my_hand)
    len_ini_hand = 12 - env.num_players

    pudines = [obs[(env.diff_num_cards + env.played_cards_size) * i + env.diff_num_cards] for i in range(env.num_players)]
    avg_pudines = sum(pudines) / env.num_players

    # Si tengo wasabi en la mano (pero no jugado) y quedan al menos 5 jugadas --> wasabi
    if my_hand[WASABI] and not my_played[WASABI] and len_hand > 4:
        return WASABI

    # Si hay Nigiri de Calamar --> Nigiri de Calamar
    if my_hand[NIGIRI_CALAMAR]: 
        return NIGIRI_CALAMAR
    
    # Tengo Wasabi --> Nigiris
    if my_played[WASABI] and my_hand[NIGIRI_SALMON]:
        return NIGIRI_SALMON
    if my_played[WASABI] and my_hand[NIGIRI_TORTILLA]:
        return NIGIRI_TORTILLA
    
    # Si hay palillos el la mano (no los he jugado) y quedan más de la mitad de rondas --> Palillos
    if my_hand[PALILLOS] and not my_played[PALILLOS] and len_hand > len_ini_hand // 2:
        return PALILLOS
    
    # Palillos Jugados + 2 tempuras en mano --> Tempura
    if my_played[PALILLOS] and my_hand[TEMPURA] >= 2:
        return TEMPURA
    # Si he jugado Tempura y hay Tempura en mano --> Tempura
    if my_played[TEMPURA] and my_hand[TEMPURA]:
        return TEMPURA
    # Quedarse por encima de la media de pudines si es posible
    if my_hand[PUDIN] and my_played[0] <= avg_pudines:
        return PUDIN
    # Selección de cartas sin palillos o con palillos pero quedan menos de la mitad de las cartas (deshacerme de los palillos)
    #  --> Coger cartas del ranking
    if not env.state_chopsticks or len_hand < len_ini_hand // 2:
        for card in CARDS_SCORE_RANKING:
            if my_hand[card]:
                return card
    else:
        # Si no, seguir el top 7 del ranking. En caso de no encontrar ninguna carta del top disponible seleccionamos una carta inválida, 
        # indicando que no quiero utilizar la carta de los palillos. 
        for card in CARDS_SCORE_RANKING[:7]:
            if my_hand[card]:
                return card
        
        # Random choice
        for card in CARDS_SCORE_RANKING:
            if not my_hand[card]:
                return card
        
def test_rules(env: SushiGoEnv, render_mode):
    # Obtain number of episodes/games to play
    episodes = input("Enter episode number, or games to play (q to quit): ").strip()
    if episodes == 'q': return False
    if episodes == '0': return True
    episodes = int(episodes) if episodes.isnumeric() else 1000

    wins = 0
    for episode in range(1, episodes + 1): 
        obs, _ = env.reset() # --> (observation, info)
        ended = False
        score = 0
        while not ended:
            reward = 0
            # SushiGo!: render the last state
            if sum(env.players[0].hand) == sum(env.players[1].hand): 
                env.render(render_mode) 
            action = rule_action_selection(env, obs) # --> (action, _states)
            obs, reward, done, truncated, info = env.step(action) 
            score += reward
            ended = done or truncated 
        env.render(render_mode) 
        env.render_scores()

        winner_ind = env.winner_indices()
        if 0 in winner_ind and len(winner_ind) == 1: wins += 1

    print(f'Win rate {wins}/{episodes}: {wins/episodes:.2f}')
        
    env.close()
    return True

def train_agent_params(model_P_ep_save_freq, model_path, episodes, save_freq, algorithm_name, num_players, log_path = None):
    # Create the directories if they do not exist
    os.makedirs(model_path, exist_ok=True)
    if log_path: os.makedirs(log_path, exist_ok=True)

    ModelClass = algorithm_dict[algorithm_name]
    name = algorithm_name + '_model'

    env = SushiGoEnv(ModelClass=ModelClass, num_players=num_players)
    # Create callback
    checkpoint_callback = LastCheckpointCallback(
        env = env, 
        save_freq_ep = save_freq, 
        max_episodes = episodes, 
        save_path= model_path, 
        name=name,
        model_P_ep_save_freq= model_P_ep_save_freq
        )
    # Calculate maximum number of timesteps
    max_timesteps = 60 * episodes # 10 cards * 3 rounds *  2 (sushigo option)
    model = ModelClass('MlpPolicy', env, verbose = 0) # or verbose = 1
    # Do the actual training
    model.learn(total_timesteps=max_timesteps , callback=checkpoint_callback)
    env.close()

# Application
if __name__ == "__main__":
    print("Welcome to the program! (by Jaime Espel)\n")
    while True:
        print("Please select an option:")
        print("  1. Check Environment Validity")
        print("  2. Train Model (Timed)")
        print("  3. Train All Algorithms")
        print("  4. Delete All Trained agents (Removes results from option 3)")
        print("  5. Apply Transfer Learning (Between environments with different player counts)")
        print("  6. Remove All Transfer Learning Agents")
        print("  7. Test RL Agents")
        print("  8. Test Rule-Based Agents")
        print("To exit, enter any other input.")
        choice = input("Enter your choice (1 - 8): ")
        match choice:
            case '1': 
                # Check environment validity 
                print('\n\tChecking environment validity\n')
                env = SushiGoEnv(num_players=2)
                check_env(env)
                print('\n\tThe environment is valid!')

            case '2':
                # TrainModel-Timed
                print('\n\tTrain a Model\n')
                import time
                # Train a model 
                num_players = ask_for_num_players()
                ini = time.time()
                train_agent(num_players)
                # Training time
                duracion = time.time() - ini
                h = int(duracion // 3600)
                mins = int((duracion % 3600) // 60)
                seg = int(duracion) % 60
                print(f"\nTotal time of training: {h} h - {mins} min - {seg} seg")


            case '3':
                # TrainAllAlgorithms
                print('\n\tTrain with every available algorithm\n')
                
                # Number of players
                num_players = ask_for_num_players() 

                # Total number of episodes (games) to train
                total_episodes = int(input("Enter total episodes/games of training: "))

                # Frequency for saving past versions of the model as opponents
                save_freq = int(input("Enter save-frequency of the model as training opponents (in episodes/games): "))

                # Intermediate models saving
                has_intermediate = input("Do you want to save intermediate models? (y/n): ")
                if has_intermediate.lower() == 'y' or has_intermediate.lower() == 'yes':
                    model_P_ep_save_freq = int(input("Enter save-frequency of intermediate models (in episodes/games): "))
                else: model_P_ep_save_freq = -1

                # algorithm_names = ['A2C', 'DQN', 'PPO', 'QR-DQN', 'TRPO']
                algorithm_names = list(algorithm_dict.keys()) 
                
                # Model dirs
                model_path = os.path.join('Training', f'P{num_players}')
                os.makedirs(model_path, exist_ok=True)
                # Algorithm loop
                for algorithm_name in algorithm_names:
                    train_agent_params(model_P_ep_save_freq, os.path.join(model_path, algorithm_name), total_episodes, save_freq, algorithm_name = algorithm_name, num_players = num_players)
                print()

            
            case '4':
                # RemoveAllTrainedLeagues
                num_players = ask_for_num_players() 
                import shutil
                directory_to_remove = os.path.join('Training', f'P{num_players}')
                if os.path.exists(directory_to_remove): shutil.rmtree(directory_to_remove)
                print(f'\n\tAll Trained agents in {directory_to_remove} deleted\n')
            
            case '5':
                # Transfer Learning

                # Choose model
                model_path_name, ModelClass, algorithm_name, from_num_players, prev_episode_number = obtain_model_info()

                # Number of players
                while not (user_input := input("Enter number of players of the new environment: ")).isdigit() or not (2 <= (to_num_players := int(user_input)) <= 5) :
                    print("Invalid input. Please enter a number [2, 5].")

                # Total number of episodes (games) to train
                total_episodes = int(input("Enter total episodes/games of training: "))

                # Frequency for saving past versions of the model as opponents
                save_freq = int(input("Enter save-frequency of the model as training opponents (in episodes/games): "))

                # Intermediate models saving
                has_intermediate = input("Do you want to save intermediate models? (y/n): ")
                if has_intermediate.lower() == 'y' or has_intermediate.lower() == 'yes':
                    model_P_ep_save_freq = int(input("Enter save-frequency of intermediate models (in episodes/games): "))
                else: model_P_ep_save_freq = -1

                # Result dir
                result_path = os.path.join('Training', 'TransferLearning', algorithm_name)
                os.makedirs(result_path, exist_ok=True)

                # Create environment
                env = SushiGoEnv(ModelClass=ModelClass, num_players=to_num_players)
                model = ModelClass.load(model_path_name, env=DummyVecEnv([lambda: env]), verbose = 0)        
                base_name = algorithm_name + f'_model_P{from_num_players}_to'
                checkpoint_callback = LastCheckpointCallback(
                        env = env, 
                        save_freq_ep = save_freq, 
                        max_episodes = total_episodes, 
                        save_path= result_path, 
                        name=base_name,
                        prev_episode_number=prev_episode_number,
                        model_P_ep_save_freq=model_P_ep_save_freq
                    ) 
                # Train
                max_timesteps = 60 * total_episodes # 10 cards * 3 rounds *  2 (sushigo option)
                model.learn(total_timesteps=max_timesteps , callback=checkpoint_callback)
                env.close()


            case '6':
                # RemoveAllTransferLearning
                import shutil
                directory_to_remove = os.path.join('Training', 'TransferLearning')
                if os.path.exists(directory_to_remove): shutil.rmtree(directory_to_remove)
                print(f'\n\tAll Trained agents in {directory_to_remove} deleted\n')
            case '7': 
                # Test RL agent(s)
                num_players = ask_for_num_players() 
                render_mode = ask_render_mode()
                print(f'\nnum_players: {num_players}')
                
                # Create environment 
                env = SushiGoEnv(num_players=num_players)
                env.players[0].player_mode = "RLAgent"
                
                # Obtain the model for P0
                try: model = obtain_model(env)
                except FileNotFoundError: print('\nNo models available!')
                else:
                    # Choose opponent modes
                    print(f'Choose opponent modes:')
                    print(f'   1. All Random')
                    print(f'   2. All Rules')
                    print(f'   Else: Customized (including humans and other models)')
                    opt = input("Enter your choice: ").strip()
                    match opt:
                        case '1': 
                            for i in range(1, env.num_players):
                                env.players[i].player_mode = 'random'
                        case '2': 
                            for i in range(1, env.num_players):
                                env.players[i].player_mode = 'RuleAgent'
                        case _: 
                            for i in range(1, env.num_players):
                                choose_opponent_mode(env, i)
                    print()

                    while test_agent(env, model, render_mode): pass # When testing P0 is always a RL agent
            
            case '8':
                # Test rule agent(s)
                num_players = ask_for_num_players()
                render_mode = ask_render_mode()
                print(f'\nnum_players: {num_players}')

                # Create environment
                env = SushiGoEnv(num_players=num_players)
                # Player modes
                env.players[0].player_mode = "rules"

                # Choose opponent modes
                print(f'Choose opponent modes:')
                print(f'   1. All Random')
                print(f'   2. All Rules')
                print(f'   Else: Customized opponent modes')
                opt = input("Enter your choice: ").strip()
                match opt:
                    case '1': 
                        for i in range(1, env.num_players):
                            env.players[i].player_mode = 'random'
                    case '2': 
                        for i in range(1, env.num_players):
                            env.players[i].player_mode = 'RuleAgent'
                    case _: 
                        for i in range(1, env.num_players):
                            choose_opponent_mode(env, i)

                while test_rules(env, render_mode): pass

            case _:
                break
        print()