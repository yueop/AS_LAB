import json

#저장한 JSON 파일 불러오기
file_path = '모델 추가 전 실험1(50).json'

with open(file_path, 'r', encoding='utf-16') as f:
    data = json.load(f)

#점수 저장할 빈 리스트 생성
dsc_scores = []
iou_scores = []

#50개 데이터에서 점수만 추출
for item in data:
    dsc_scores.append(item['metrics']['dsc'])
    iou_scores.append(item['metrics']['iou'])

#평균 계산
avg_dsc = sum(dsc_scores) / len(dsc_scores)
avg_iou = sum(iou_scores) / len(iou_scores)

print(f"총 평가 이미지 수: {len(data)}장")
print(f"심장 분할 평균 DSC: {avg_dsc:.4f}")
print(f"심장 분할 평균 IoU: {avg_iou:.4f}")