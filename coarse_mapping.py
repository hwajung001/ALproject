import numpy as np
# class 바깥에 정의 (유틸 함수)
def get_cifar100_fine_to_coarse_dict():
		"""
		CIFAR-100 fine(0~99) → coarse(0~19) 매핑 딕셔너리 반환
		"""
		return {
				4: 0, 30: 0, 55: 0, 72: 0, 95: 0,
				1: 1, 32: 1, 67: 1, 73: 1, 91: 1,
				54: 2, 62: 2, 70: 2, 82: 2, 92: 2,
				9: 3, 10: 3, 16: 3, 28: 3, 61: 3,
				0: 4, 51: 4, 53: 4, 57: 4, 83: 4,
				22: 5, 39: 5, 40: 5, 86: 5, 87: 5,
				5: 6, 20: 6, 25: 6, 84: 6, 94: 6,
				6: 7, 7: 7, 14: 7, 18: 7, 24: 7,
				3: 8, 42: 8, 43: 8, 88: 8, 97: 8,
				12: 9, 17: 9, 37: 9, 68: 9, 76: 9,
				23: 10, 33: 10, 49: 10, 60: 10, 71: 10,
				15: 11, 19: 11, 21: 11, 31: 11, 38: 11,
				34: 12, 63: 12, 64: 12, 66: 12, 75: 12,
				26: 13, 45: 13, 77: 13, 79: 13, 99: 13,
				2: 14, 11: 14, 35: 14, 46: 14, 98: 14,
				27: 15, 29: 15, 44: 15, 78: 15, 93: 15,
				36: 16, 50: 16, 65: 16, 74: 16, 80: 16,
				8: 17, 13: 17, 48: 17, 58: 17, 90: 17,
				41: 18, 52: 18, 56: 18, 59: 18, 96: 18,
				47: 19, 69: 19, 81: 19, 85: 19, 89: 19
		}

#validation weight 구하기 위해
def compute_fine_class_accuracy(pred_fine, true_fine, num_classes=100):
    correct = np.zeros(num_classes)
    total = np.zeros(num_classes)

    for pf, tf in zip(pred_fine, true_fine):
        total[tf] += 1
        if pf == tf:
            correct[tf] += 1

    acc = correct / (total + 1e-9)
    acc = acc / np.max(acc)
    return acc


def compute_coarse_accuracy(preds, targets):
    correct = np.sum(preds == targets)
    return correct / len(targets)

def print_prediction_examples(pred_fine, true_fine, pred_coarse, true_coarse, n=10):
    """
    fine label과 coarse label 예측을 모두 출력하는 함수
    - pred_fine: 예측 fine label (N,)
    - true_fine: 정답 fine label (N,)
    - pred_coarse: 예측 coarse label (N,)
    - true_coarse: 정답 coarse label (N,)
    - n: 출력할 샘플 개수
    """
    for i in range(min(n, len(pred_fine))):
        print(f"[{i:02d}] predict fine label:    {pred_fine[i]}")
        print(f"     true fine label:      {true_fine[i]}")
        print(f"     predict coarse label: {pred_coarse[i]}")
        print(f"     true coarse label:    {true_coarse[i]}\n")

class CoarseMapper:
		def __init__(self, fine_to_coarse: dict):
				self.f2c = fine_to_coarse
				self.num_coarse = max(fine_to_coarse.values()) + 1

		def argmax_mapping(self, fine_probs, return_probs=False):
			"""
			방식 1: softmax에서 가장 확률 높은 fine-class → coarse-class로 매핑
			fine_probs: (N, 100) softmax 결과
			return_probs: True면 one-hot coarse softmax, False면 class index 반환
			"""
			fine_preds = np.argmax(fine_probs, axis=1)
			coarse_preds = np.array([self.f2c[f] for f in fine_preds])

			if return_probs:
				coarse_probs = np.zeros((len(coarse_preds), self.num_coarse))
				coarse_probs[np.arange(len(coarse_preds)), coarse_preds] = 1.0
				return coarse_probs  # shape (N, 20) one-hot
			else:
				return coarse_preds  # shape (N,)
			

		def entropy_weighted_mapping(self, fine_probs, return_probs=False):
			"""
			방식 2: 엔트로피 기반 confidence로 coarse-class softmax 생성 후 예측
			fine_probs: (N, 100)
			return_probs: True면 coarse softmax 전체 반환, False면 argmax된 coarse class 반환
			"""
			N = fine_probs.shape[0]
			log100 = np.log(100)
			coarse_probs = np.zeros((N, self.num_coarse))

			for i in range(N):
				probs = fine_probs[i]
				entropy = -np.sum(probs * np.log(probs + 1e-9))
				confidence = 1 - (entropy / log100)
				weighted_probs = probs * confidence

				for fine_idx, prob in enumerate(weighted_probs):
					coarse_idx = self.f2c[fine_idx]
					coarse_probs[i][coarse_idx] += prob

			if return_probs:
				return coarse_probs  # shape (N, 20)
			else:
				return np.argmax(coarse_probs, axis=1)  # shape (N,)
			

		def soft_average_mapping(self, fine_probs, return_probs=False):
			"""
			방식 3: fine-class softmax를 coarse-class 단위로 합산하여 coarse softmax 생성
			fine_probs: (N, 100)
			return_probs: True면 coarse softmax (N, 20), False면 argmax된 coarse class (N,)
			"""
			batch_size = fine_probs.shape[0]
			coarse_probs = np.zeros((batch_size, self.num_coarse))

			for fine_idx in range(100):
				coarse_idx = self.f2c[fine_idx]
				coarse_probs[:, coarse_idx] += fine_probs[:, fine_idx]

			if return_probs:
				return coarse_probs
			else:
				return np.argmax(coarse_probs, axis=1)	


		def softlabel_coarse_mapping(fine_probs, fine_to_coarse_dict, num_coarse=20):
			batch_size = fine_probs.shape[0]
			coarse_probs = np.zeros((batch_size, num_coarse))
			for fine_idx in range(100):
				coarse_idx = fine_to_coarse_dict[fine_idx]
				coarse_probs[:, coarse_idx] += fine_probs[:, fine_idx]
			return coarse_probs  # shape (N, 20)

			


		def validation_guided_mapping(self, fine_probs, fine_weights, return_probs=False):
			"""
			방식 4: fine-class validation accuracy를 weight로 곱한 후 coarse-class로 매핑
			fine_probs: (N, 100) softmax 결과
			fine_weights: (100,) validation accuracy 기반 가중치 (0~1 정규화)
			return_probs: True면 coarse softmax, False면 argmax coarse class
			"""
			weighted_fine = fine_probs * fine_weights[np.newaxis, :]  # shape (N, 100)
			coarse_probs = np.zeros((fine_probs.shape[0], self.num_coarse))

			for f in range(100):
				c = self.f2c[f]
				coarse_probs[:, c] += weighted_fine[:, f]

			if return_probs:
				return coarse_probs  # shape (N, 20)
			else:
				return np.argmax(coarse_probs, axis=1)  # shape (N,)
