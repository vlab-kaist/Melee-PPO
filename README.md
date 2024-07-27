DONE:
- skrl Agent customize 진행: Agent act() 수정
- gamestate 직접 입력 받아 preprocess 적용
- op agent 학습없이 동작만 하는 env 구축 (SelfPlayMeleeEnv)
- GRU Actor, Critic 모델 적용 => 학습 잘 안됨 => 호수님 TransformerGRU 구현
- frame stack env 구축, gamestate 받아서 obs stack 하는 agent 구현
- 안끊기게 safe learning 환경 구축 (logging 이어서 진행)
- league based self play 구현 (초안, 개선 필요)

WIP:
- state에 projectile 추가 후 6개 agent cpu 학습

TODO:
- Platform 고려 state 수정
