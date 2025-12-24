import pygame
import cv2
import mediapipe as mp
import numpy as np

# Inicializar MediaPipe y Pygame
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
pygame.init()

# Dimensiones
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Juego rompebloques ejecutando")
clock = pygame.time.Clock()

# Paddle
paddel = pygame.image.load('padel.png')
paddel = pygame.transform.scale(paddel, (150, 40))
paddelx, paddely = WIDTH // 2, HEIGHT - 100
padde_widht, paddel_height = 150,40
paddel_speed = 10

# Pelota
ball = pygame.image.load('balon-removebg-preview.png')
ball = pygame.transform.scale(ball, (40, 40))
ballx, bally = WIDTH // 2, HEIGHT // 2
ball_speed_x, ball_speed_y = 20, 20
ball_radius = 10

# Bloques
bloque_img = pygame.image.load('bloque-removebg-preview.png')
bloque_rows, bloque_columns = 4, 8
bloque_width, bloque_height = WIDTH // bloque_columns, 50
bloque_img = pygame.transform.scale(bloque_img, (bloque_width, bloque_height))

bloques = []
for i in range(bloque_rows):
    for col in range(bloque_columns):
        bloques.append(pygame.Rect(col * bloque_width, i * bloque_height, bloque_width, bloque_height))

# Cámara
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Mediapipe manos
with mp_hands.Hands(min_detection_confidence=0.5,
                    min_tracking_confidence=0.5,
                    max_num_hands=1) as hands:

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        ret, frame = cap.read()
        if not ret:
            continue

        # Procesar frame para MediaPipe
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                indice_finger = hand_landmarks.landmark[8]
                paddelx = int(indice_finger.x * WIDTH) - padde_widht // 2
                mp_draw.draw_landmarks(rgb_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Convertir frame a superficie para Pygame
        # Asegúrate de convertirlo a RGB para que sea compatible con Pygame
        rgb_frame = np.array(rgb_frame)
        cam_surface = pygame.surfarray.make_surface(rgb_frame.swapaxes(0, 1))
        cam_surface = pygame.transform.scale(cam_surface, (WIDTH, HEIGHT))
        screen.blit(cam_surface, (0, 0))  # Fondo de cámara

        # Mover pelota
        ballx += ball_speed_x
        bally += ball_speed_y

        if ballx -ball_radius <0 or ballx + ball_radius> WIDTH:
            ball_speed_x *= -1


        if bally -ball_radius <0 :
            ball_speed_y *= -1

        if bally + ball_radius >HEIGHT:
            running = False


        if paddelx<0:
            paddelx = 0

        if paddelx +padde_widht>WIDTH:
            paddelx = WIDTH-padde_widht



        if paddelx<ballx <paddelx +padde_widht and bally + ball_radius >= paddely:
            ball_speed_y *= -1


        for i in bloques:
            if i.collidepoint(ballx,bally):
                bloques.remove((i))
                ball_speed_y *=-1
                break

        # Mostrar bloques
        for bloque in bloques:
            screen.blit(bloque_img, (bloque.x, bloque.y))

        # Mostrar paddle y pelota
        screen.blit(paddel, (paddelx, paddely))
        screen.blit(ball, (ballx - ball_radius, bally - ball_radius))

        pygame.display.update()
        clock.tick(30)

cap.release()
cv2.destroyAllWindows()
pygame.quit()
