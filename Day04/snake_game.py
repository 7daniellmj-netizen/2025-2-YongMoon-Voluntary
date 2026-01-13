import numpy as np
from collections import deque

class SnakeEnv:
    def __init__(self, render_mode=None, target_fps=10, grid_size=20):
        """
        환경 초기화
        
        Args:
            render_mode: 'human'이면 화면 렌더링, None이면 헤드리스 모드
            target_fps: 게임 속도 (FPS)
            grid_size: 그리드 크기 (기본 20x20)
        """
        # 게임 설정
        self.grid_size = grid_size  # 그리드 크기 (커스터마이징 가능)
        self.cell_size = 30  # 각 셀의 픽셀 크기 (커스터마이징 가능)
        
        self.width = self.grid_size * self.cell_size
        self.height = self.grid_size * self.cell_size

        # 뱀 초기 설정
        self.snake = None  # 뱀의 몸통 좌표 리스트 [(x, y), ...]
        self.direction = None  # 현재 이동 방향: 0=위, 1=오른쪽, 2=아래, 3=왼쪽
        
        # 먹이 위치
        self.food_x = None
        self.food_y = None
        
        # 게임 상태
        self.score = 0
        self.steps = 0
        self.max_steps = grid_size * grid_size * 100  # 최대 스텝 (무한 루프 방지)

        # 액션 스페이스와 관측 스페이스 정의
        self.action_space_n = 4  # 0: 위, 1: 오른쪽, 2: 아래, 3: 왼쪽
        self.observation_space_shape = (11,)  # 관측값 개수
        
        # 렌더링 설정
        self.render_mode = render_mode
        self.target_fps = target_fps

        # 화면 렌더링 변수
        self.screen = None
        self.clock = None
        self.font = None
        self.small_font = None
        
        # 색상 설정 (커스터마이징 가능)
        self.COLOR_BG = (20, 20, 40)  # 배경색
        self.COLOR_SNAKE_HEAD = (0, 255, 0)  # 뱀 머리 색
        self.COLOR_SNAKE_BODY = (0, 200, 0)  # 뱀 몸통 색
        self.COLOR_FOOD = (255, 50, 50)  # 먹이 색
        self.COLOR_GRID = (40, 40, 60)  # 그리드 선 색
        
        if render_mode == 'human':
            try:
                import pygame
                pygame.init()
                self.screen = pygame.display.set_mode((self.width, self.height))
                pygame.display.set_caption("Snake 게임")
                self.clock = pygame.time.Clock()
                self.font = pygame.font.Font(None, 74)
                self.small_font = pygame.font.Font(None, 36)
            except ImportError:
                print("pygame이 설치되지 않았습니다. headless mode로 실행하거나 pygame을 설치하세요.")
                self.render_mode = None

    def reset(self):
        """환경을 초기 상태로 리셋"""
        # 뱀을 화면 중앙에서 시작 (길이 3)
        center = self.grid_size // 2
        self.snake = deque([
            (center, center),
            (center, center + 1),
            (center, center + 2)
        ])
        
        # 초기 방향: 위쪽
        self.direction = 0
        
        # 먹이 생성
        self._spawn_food()
        
        # 게임 상태 초기화
        self.score = 0
        self.steps = 0

        return self._get_state()

    def _spawn_food(self):
        """
        TODO 1: 먹이 생성 로직 구현
        뱀의 몸통과 겹치지 않는 위치에 먹이를 랜덤하게 생성
        
        힌트:
        - np.random.randint(0, self.grid_size)로 랜덤 위치 생성
        - while 루프로 뱀과 겹치지 않을 때까지 반복
        - (x, y) in self.snake로 겹침 체크
        """
        while True:
            x = np.random.randint(0, self.grid_size)
            y = np.random.randint(0, self.grid_size)
            if (x, y) not in self.snake:
                self.food_x = x
                self.food_y = y
                break

    def step(self, action):
        """
        action:
        0 = 위로 이동
        1 = 오른쪽으로 이동
        2 = 아래로 이동
        3 = 왼쪽으로 이동

        반환값:
        state : 현재 상태
        reward : 보상
        done : 에피소드 종료 여부
        info : 추가 정보 (딕셔너리)
        """
        self.steps += 1

        # TODO 2: 방향 전환 로직 구현
        # 반대 방향으로는 이동할 수 없음
        # 예: 현재 위(0)로 가고 있으면 아래(2)로 변경 불가
        # hint: (self.direction + 2) % 4 != action 조건 사용
        
        # 방향 업데이트 (반대 방향은 무시)
        if (self.direction + 2) % 4 != action:
            self.direction = action

        # TODO 3: 뱀 머리의 새로운 위치 계산
        # 현재 방향(self.direction)에 따라 머리 위치 이동
        # 0=위(y-1), 1=오른쪽(x+1), 2=아래(y+1), 3=왼쪽(x-1)
        
        # 현재 머리 위치
        head_x, head_y = self.snake[0]
        
        # 방향에 따라 새로운 머리 위치 계산
        if self.direction == 0:  # 위
            new_head = (head_x, head_y - 1)
        elif self.direction == 1:  # 오른쪽
            new_head = (head_x + 1, head_y)
        elif self.direction == 2:  # 아래
            new_head = (head_x, head_y + 1)
        else:  # 왼쪽 (3)
            new_head = (head_x - 1, head_y)

        reward = 0.0
        done = False

        # TODO 4: 충돌 감지 구현
        # 1. 벽 충돌: 새 머리 위치가 그리드 범위를 벗어남
        # 2. 자기 몸통 충돌: 새 머리 위치가 뱀의 몸통에 있음
        # hint: new_head in self.snake로 몸통 충돌 체크
        
        # 벽 충돌 체크
        if (new_head[0] < 0 or new_head[0] >= self.grid_size or
            new_head[1] < 0 or new_head[1] >= self.grid_size):
            reward = -10.0
            done = True
            info = {'score': self.score, 'reason': 'wall_collision'}
            return self._get_state(), reward, done, info
        
        # 자기 몸통 충돌 체크
        if new_head in self.snake:
            reward = -10.0
            done = True
            info = {'score': self.score, 'reason': 'self_collision'}
            return self._get_state(), reward, done, info

        # TODO 5: 뱀 이동 및 먹이 먹기 로직 구현
        # 1. 새 머리를 뱀의 맨 앞에 추가 (self.snake.appendleft)
        # 2. 먹이를 먹었는지 체크 (new_head == (self.food_x, self.food_y))
        #    - 먹었으면: score 증가, reward=10, 새 먹이 생성
        #    - 안 먹었으면: 꼬리 제거 (self.snake.pop)
        
        # 새 머리 추가
        self.snake.appendleft(new_head)
        
        # 먹이를 먹었는지 체크
        if new_head[0] == self.food_x and new_head[1] == self.food_y:
            # 먹이를 먹음
            self.score += 1
            reward = 10.0
            self._spawn_food()
        else:
            # 먹이를 안 먹음 - 꼬리 제거
            self.snake.pop()
            reward = -0.01  # 작은 페널티 (시간이 지날수록)

        # 최대 스텝 도달 체크
        if self.steps >= self.max_steps:
            done = True

        info = {
            'score': self.score,
            'steps': self.steps,
            'snake_length': len(self.snake)
        }

        return self._get_state(), reward, done, info

    def _get_state(self):
        """
        TODO 6: AI에게 줄 입력값(관측값) 추출
        
        반환할 값들 (11개):
        1-4. 위험 감지 (4방향): 위, 오른쪽, 아래, 왼쪽에 장애물이 있는지
        5-8. 먹이 방향 (4방향): 먹이가 위, 오른쪽, 아래, 왼쪽에 있는지
        9-11. 현재 방향 (원-핫 인코딩 중 3개만): 위, 오른쪽, 아래
        
        hint: 
        - 장애물 = 벽 or 자기 몸통
        - 먹이 방향 = 머리 기준으로 먹이의 상대 위치
        """
        head_x, head_y = self.snake[0]
        
        # 4방향 위험 체크
        danger_up = self._is_collision(head_x, head_y - 1)
        danger_right = self._is_collision(head_x + 1, head_y)
        danger_down = self._is_collision(head_x, head_y + 1)
        danger_left = self._is_collision(head_x - 1, head_y)
        
        # 먹이 방향
        food_up = float(self.food_y < head_y)
        food_right = float(self.food_x > head_x)
        food_down = float(self.food_y > head_y)
        food_left = float(self.food_x < head_x)
        
        # 현재 방향 (원-핫 인코딩, 4개 중 3개만 사용)
        dir_up = float(self.direction == 0)
        dir_right = float(self.direction == 1)
        dir_down = float(self.direction == 2)
        # dir_left는 위 3개로 추론 가능하므로 생략
        
        return np.array([
            danger_up, danger_right, danger_down, danger_left,
            food_up, food_right, food_down, food_left,
            dir_up, dir_right, dir_down
        ], dtype=np.float32)

    def _is_collision(self, x, y):
        """
        주어진 위치에 장애물이 있는지 체크
        장애물 = 벽 or 뱀의 몸통
        """
        # 벽 체크
        if x < 0 or x >= self.grid_size or y < 0 or y >= self.grid_size:
            return 1.0
        # 뱀 몸통 체크
        if (x, y) in self.snake:
            return 1.0
        return 0.0

    def render(self):
        """
        게임 화면 렌더링 (render_mode='human'일 때만 작동)
        """
        if self.render_mode != 'human' or self.screen is None:
            return
        
        import pygame
        
        # 배경
        self.screen.fill(self.COLOR_BG)
        
        # 그리드 선 그리기 (선택사항)
        for x in range(0, self.width, self.cell_size):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.height))
        for y in range(0, self.height, self.cell_size):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.width, y))
        
        # 먹이 그리기
        food_rect = pygame.Rect(
            self.food_x * self.cell_size,
            self.food_y * self.cell_size,
            self.cell_size,
            self.cell_size
        )
        pygame.draw.rect(self.screen, self.COLOR_FOOD, food_rect, border_radius=5)
        
        # 뱀 그리기
        for i, (x, y) in enumerate(self.snake):
            rect = pygame.Rect(
                x * self.cell_size + 2,
                y * self.cell_size + 2,
                self.cell_size - 4,
                self.cell_size - 4
            )
            if i == 0:  # 머리
                pygame.draw.rect(self.screen, self.COLOR_SNAKE_HEAD, rect, border_radius=8)
            else:  # 몸통
                pygame.draw.rect(self.screen, self.COLOR_SNAKE_BODY, rect, border_radius=5)
        
        # 점수 표시
        score_text = self.small_font.render(f"Score: {self.score}", True, (255, 255, 255))
        self.screen.blit(score_text, (10, 10))
        
        # 길이 표시
        length_text = self.small_font.render(f"Length: {len(self.snake)}", True, (255, 255, 255))
        self.screen.blit(length_text, (10, 45))
        
        pygame.display.flip()
        
        if self.clock:
            self.clock.tick(self.target_fps)
    
    def render_game_over(self):
        """
        게임 오버 화면 렌더링
        
        Returns:
            button_rect: 재시작 버튼의 영역 (pygame.Rect 객체)
        """
        if self.render_mode != 'human' or self.screen is None:
            return None
        
        import pygame
        
        # 반투명 오버레이
        overlay = pygame.Surface((self.width, self.height))
        overlay.set_alpha(180)
        overlay.fill((0, 0, 0))
        self.screen.blit(overlay, (0, 0))
        
        # GAME OVER 텍스트
        game_over_font = pygame.font.Font(None, 100)
        game_over_text = game_over_font.render("GAME OVER", True, (255, 50, 50))
        text_rect = game_over_text.get_rect(center=(self.width // 2, self.height // 2 - 100))
        self.screen.blit(game_over_text, text_rect)
        
        # 최종 점수 표시
        score_font = pygame.font.Font(None, 50)
        score_text = score_font.render(f"Final Score: {self.score}", True, (255, 255, 255))
        score_rect = score_text.get_rect(center=(self.width // 2, self.height // 2 - 20))
        self.screen.blit(score_text, score_rect)
        
        # 최종 길이 표시
        length_text = score_font.render(f"Snake Length: {len(self.snake)}", True, (255, 255, 255))
        length_rect = length_text.get_rect(center=(self.width // 2, self.height // 2 + 30))
        self.screen.blit(length_text, length_rect)
        
        # 다시 시작 버튼
        button_width = 250
        button_height = 60
        button_x = self.width // 2 - button_width // 2
        button_y = self.height // 2 + 90
        button_rect = pygame.Rect(button_x, button_y, button_width, button_height)
        
        # 마우스 호버 체크
        mouse_pos = pygame.mouse.get_pos()
        is_hover = button_rect.collidepoint(mouse_pos)
        
        # 버튼 모양
        button_color = (100, 255, 100) if is_hover else (50, 200, 50)
        pygame.draw.rect(self.screen, button_color, button_rect, border_radius=15)
        pygame.draw.rect(self.screen, (255, 255, 255), button_rect, 4, border_radius=15)
        
        # 버튼 텍스트
        button_font = pygame.font.Font(None, 48)
        button_text = button_font.render("RESTART", True, (0, 0, 0))
        button_text_rect = button_text.get_rect(center=button_rect.center)
        self.screen.blit(button_text, button_text_rect)
        
        pygame.display.flip()
        
        return button_rect

    def close(self):
        """환경 종료 및 리소스 정리"""
        if self.screen is not None:
            import pygame
            pygame.quit()
            self.screen = None
            self.clock = None


# 테스트 코드
if __name__ == "__main__":
    import pygame
    
    # 렌더링 모드로 환경 생성
    # 커스터마이징 가능 파라미터:
    # - target_fps: 게임 속도 (낮을수록 느림)
    # - grid_size: 그리드 크기 (작을수록 쉬움)
    env = SnakeEnv(render_mode='human', target_fps=10, grid_size=20)
    
    # 게임 초기화
    state = env.reset()
    print(f"초기 상태: {state}")
    print(f"상태 형태: {state.shape}")
    print(f"액션 스페이스: {env.action_space_n}")
    print("\n조작법: 화살표 키로 방향 전환")
    print("- 위 화살표: 위로")
    print("- 오른쪽 화살표: 오른쪽으로")
    print("- 아래 화살표: 아래로")
    print("- 왼쪽 화살표: 왼쪽으로")
    
    # 게임 플레이 (키보드 조작)
    running = True
    total_reward = 0
    steps = 0
    action = 0  # 초기 방향: 위
    
    while running:
        # Pygame 이벤트 처리
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            # 키보드 이벤트로 즉시 방향 전환
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    action = 0
                elif event.key == pygame.K_RIGHT:
                    action = 1
                elif event.key == pygame.K_DOWN:
                    action = 2
                elif event.key == pygame.K_LEFT:
                    action = 3
        
        # 스텝 실행
        next_state, reward, done, info = env.step(action)
        total_reward += reward
        steps += 1
        
        # 렌더링
        env.render()
        
        # 상태 정보 출력
        if steps % 50 == 0:
            print(f"Steps: {steps}, Reward: {reward:.2f}, Total: {total_reward:.2f}")
            print(f"Score: {info['score']}, Length: {info['snake_length']}")
        
        # 게임 종료 시 리셋
        if done:
            # GAME OVER 화면 표시 및 버튼 대기
            waiting = True
            while waiting:
                button_rect = env.render_game_over()
                
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                        waiting = False
                    # 마우스 클릭 체크
                    if event.type == pygame.MOUSEBUTTONDOWN:
                        if button_rect and button_rect.collidepoint(event.pos):
                            waiting = False
            
            print(f"\n게임 종료! 총 스텝: {steps}, 총 보상: {total_reward:.2f}")
            print(f"최종 점수: {info['score']}, 최종 길이: {info.get('snake_length', len(env.snake))}")
            if 'reason' in info:
                print(f"종료 이유: {info['reason']}\n")
            
            state = env.reset()
            total_reward = 0
            steps = 0
            action = 0
    
    env.close()