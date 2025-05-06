# ALproject
CIFAR-100 Fine-to-Coarse Mapping Project

🧰 개요

이 프로젝트는 CIFAR-100 데이터셋을 대상으로, 딥러닝 라이브러리를 사용하지 않고 인공신경망을 직접 구현하여 이미지 분류기를 제작하고, 다음 두 수준의 분류 성능을 실험합니다:

파인 클래스 (Fine Class, 100개)

코어스 클래스 (Coarse Class, 상위 클래스, 20개)

fine 클래스에 대해 학습한 모델만을 사용하여, coarse 수준의 분류를 어떻게 수행할 수 있는지를 다양한 매핑 전략으로 탐색합니다.

🚀 목표

MiniVGGNet 모델을 직접 구현하여 fine class 분류기 학습

coarse class 정확도를 추정할 수 있는 매핑 기법 적용 및 비교 평가

