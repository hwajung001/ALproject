# ALproject

# CIFAR-100 Fine-to-Coarse Mapping Project

## 🧰 개요

이 프로젝트는 CIFAR-100 데이터셋을 대상으로, 딥러닝 라이브러리를 사용하지 않고 인공신경망을 직접 구현하여 이미지 분류기를 제작하고, 분류를 진행합니다.

- **파인 클래스 (Fine Class, 100개)**
- **코어스 클래스 (Coarse Class, 상위 클래스, 20개)**

fine 클래스에 대해 학습한 모델만을 사용하여, coarse 수준의 분류를 어떻게 수행할 수 있는지를 다양한 매핑 전략으로 탐색합니다.

## 🚀 목표
- MiniVGGNet 모델을 직접 구현하여 fine class 분류기 학습  
- coarse class 정확도를 추정할 수 있는 매핑 기법 적용 및 비교 평가


## 🔄 매핑 전략
### 1. Argmax 기반 매핑

- 모델의 출력값 중 가장 높은 확률을 갖는 fine 클래스 예측: `fine = argmax(predict(x))`
- fine → coarse 매핑 테이블을 통해 coarse 클래스 추론: `coarse = fine_to_coarse[fine]`

### 2. Softmax 기반 coarse 확률 계산
- 모델 출력(logit)에 softmax를 적용해 fine class 확률 분포 생성
- coarse class 별로 관련된 fine class들의 확률을 평균 또는 가중합
- 최종 coarse 확률 분포 계산: `P_k = 평균(p_i for i in C_k)`

## 📄 파일 구조

