import numpy as np

students = ["짱구", "철수", "맹구"]
scores = [85, 92, 78]

# TODO: enumerate와 zip을 조합하거나 활용하여 80점 이상 학생 정보 출력
# TODO: 예상 출력 "1번 짱구 학생 (점수: 85점)"
for idx, (name,score) in enumerate(zip(students, scores), start=1):
      if score >= 80:
        print(f"{idx}번 {name}학생 (점수: {score}점)\n")


scores = np.array([
    [83, 92, 100],  # 0번 학생
    [76, 63, 58],   # 1번 학생
    [91, 99, 84]    # 2번 학생
])

# TODO: NumPy 연산 함수를 사용하여 배열을 정규화

np_arr_normalized = (scores - np.min(scores)) / (np.max(scores) - np.min(scores))
print(np_arr_normalized)
print("")

# TODO: NumPy 연산 함수를 사용하여 배열을 0.0 ~ 1.0 범위의 float32 배열로 변환
# TODO: 최종 배열의 최솟값 = 0.0, 최종 배열의 최댓값 = 1.0

np_arr_final = np_arr_normalized.astype(np.float32)
print(np_arr_final)
print(np_arr_final.dtype)
print("")

raw_students = [
    "zg0505_짱구",
    "cs0822_철수",
    "mg0910_맹구",
    "yr0605_유리",
    "hi0205_훈이"
    ]

raw_scores = np.array([
    [95, 82, 88],   # 짱구
    [70, 65, 90],   # 철수
    [98, 100, 92],  # 맹구
    [60, 75, 80],   # 유리
    [81, 88, 84],   # 훈이
])

subjects = ["국어", "수학", "과학"]

# TODO: list comprehension을 활용해 raw_students 리스트에서 이름만 추출한 리스트 생성

student_names = [name for student in raw_students for name in student.split('_')[1:]]
print("1. 학생 명단: ", end="")
for idx, name in enumerate(student_names):
    sep = ", " if idx < len(student_names) - 1 else "\n\n"
    print(name, end=sep)


# TODO: 전산 시스템에 입력된 모든 성적들을 1차원 배열로 출력 (총 15개)
all_scores = raw_scores.flatten()
print("2. 전산 시스템에 입력된 모든 성적: ", end="")
for idx, score in enumerate(all_scores):
    sep = ", " if idx < len(all_scores) - 1 else "\n\n"
    print(f"{score}점", end=sep)

# TODO: 전체 12개 성적 중 85점 이상 개수

over_85 = (all_scores >= 85).sum()
print(f"3. 전체 15개 성적 중 85점 이상 개수: {over_85}개", end="\n\n")

# TODO: 학생별 평균 점수 출력
student_means = np.mean(raw_scores, axis=1)
print("4. 학생별 평균 점수")
for name, score in zip(student_names, student_means):
    print(f"{name}: {score}점")
print("")
# 예상 출력: "4. 학생별 평균 점수: 짱구 00점, 철수 00점, 맹구 00점, 유리 00점, 훈이 00점"


# TODO: 평균 점수가 가장 높은 학생 선택
top_student_idx = student_means.argmax()
top_student_name = student_names[top_student_idx]
print(f"5. 최고 평균 수석 학생: {top_student_name} ({student_means[top_student_idx]}점)\n")  # 예상 출력: "5. 최고 평균 수석 학생: XX (xx.xx점)"

# TODO: 전체 점수 기준, 최고 점수와 과목 그리고 해당 학생 계산 (modulo % 사용 추천)
overall_max_score = raw_scores.max()
overall_max_idx = raw_scores.argmax()
overall_max_subject = subjects[overall_max_idx % 3]
overall_max_student = student_names[overall_max_idx // 3]
print(f"6. 단일 과목 최고 점수: {overall_max_score}점 (과목:{overall_max_subject}, 학생:{overall_max_student})")  # 예상 출력: "6. 단일 과목 최고 점수: xx점 (과목: XX, 학생: XX)"

