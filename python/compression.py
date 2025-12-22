import heapq #우선순위 큐 구현을 위한 모듈
import os #파일의 크기 측정을 위해 운영체제와 상호작용하기 위한 모듈
import pickle #frequency(빈도수) 딕셔너리를 헤더의 저장하기 위한 모듈
from collections import Counter #데이터의 개수를 세 빈도수 딕셔너리를 만들기 위한 모듈

# 1. Suffix Array를 활용한 BWT(Burrows Wheeler Transform)구현
def get_suffix_array(int_list):
    """ 정수 리스트를 받아 접미사 배열(인덱스 리스트)을 반환 """
    n = len(int_list)
    # 1-1. 메모리 절약을 위해 실제 배열 복사 없이 인덱스만 정렬
    sa = sorted(range(n), key=lambda i: int_list[i:])
    return sa

def bwt_transform(data_bytes):
    """ 바이너리 데이터 -> BWT 변환된 정수 리스트 """
    # 1-2. 바이트(0~255)보다 작은 -1을 EOS(end of string)로 사용하여 정렬 시 맨 앞으로 오게 함
    int_data = list(data_bytes) + [-1]
    sa = get_suffix_array(int_data) # Suffix Array 생성
    
    # 1-3. BWT 생성
    bwt = []
    for i in sa:
        if i == 0:
            bwt.append(-1) # 맨 앞의 앞은 EOS
        else:
            bwt.append(int_data[i-1])
            
    return bwt


# 2. MTF (Move-To-Front)
def mtf_transform(int_list):
    # -1(EOS)을 포함하여 257개의 알파벳 리스트 생성
    # -1이 가장 앞에 오도록 설정 (정렬 순서 고려)
    alphabet = [-1] + list(range(256))
    result = []
    
    for val in int_list:
        idx = alphabet.index(val)
        result.append(idx)
        
        # 사용한 값을 맨 앞으로 이동
        alphabet.pop(idx)
        alphabet.insert(0, val)
        
    return result

# 3. Huffman Coding Nodes
class HuffmanNode:
    def __init__(self, char, freq):
        self.char = char
        self.freq = freq
        self.left = None
        self.right = None

    def __lt__(self, other):
        return self.freq < other.freq

# 4. BitWriter
class BitWriter:
    def __init__(self, filepath):
        self.file = open(filepath, 'wb')
        self.buffer = 0
        self.count = 0

    def write_bit(self, bit):
        # 버퍼에 비트 추가
        self.buffer = (self.buffer << 1) | bit
        self.count += 1
        
        # 8비트(1바이트)가 꽉 차면 파일에 씀
        if self.count == 8:
            self.file.write(bytes([self.buffer]))
            self.buffer = 0
            self.count = 0

    def flush(self):
        # 남은 비트가 있다면 뒤에 0을 채워서(Padding) 저장
        if self.count > 0:
            self.buffer = self.buffer << (8 - self.count)
            self.file.write(bytes([self.buffer]))
            
    def close(self):
        self.flush()
        self.file.close()

# 5. 메인 압축 클래스
class HuffmanCompressor:
    def __init__(self):
        self.codes = {} 

    def make_heap(self, frequency):
        heap = []
        for key, freq in frequency.items():
            node = HuffmanNode(key, freq)
            heapq.heappush(heap, node)
        return heap

    def merge_nodes(self, heap):
        # 힙에 노드가 하나 남을 때까지 병합
        if not heap: return None # 빈 파일 예외처리
        
        while len(heap) > 1:
            node1 = heapq.heappop(heap)
            node2 = heapq.heappop(heap)
            merged = HuffmanNode(None, node1.freq + node2.freq)
            merged.left = node1
            merged.right = node2
            heapq.heappush(heap, merged)
        return heapq.heappop(heap)

    def make_codes_helper(self, root, current_code):
        if root is None: return
        if root.char is not None:
            self.codes[root.char] = current_code
            return
        self.make_codes_helper(root.left, current_code + "0")
        self.make_codes_helper(root.right, current_code + "1")

    def compress(self, input_path, output_path):
        print(f"[{input_path}] 압축 시작")
        
        # 5-1. 파일 읽기 (바이너리 모드 'rb')
        try:
            with open(input_path, 'rb') as file:
                raw_data = file.read()
        except FileNotFoundError:
            print("파일을 찾을 수 없습니다.")
            return

        if not raw_data:
            print("빈 파일입니다.")
            return

        # 5-2. 전처리 파이프라인: BWT -> MTF
        # raw_data(bytes) -> bwt_data(int list) -> mtf_data(int list)
        print("1단계: BWT 변환")
        bwt_data = bwt_transform(raw_data)
        
        print("2단계: MTF 변환")
        mtf_data = mtf_transform(bwt_data)
        
        # 5-3. 허프만 트리 생성 (MTF 결과인 숫자 리스트를 대상으로 함)
        print("3단계: 허프만 트리 생성")
        frequency = Counter(mtf_data)
        heap = self.make_heap(frequency)
        root = self.merge_nodes(heap)
        self.make_codes_helper(root, "")

        # 5-4. 파일 저장
        print("4단계: 파일 저장 중...")
        
        # 헤더 저장: 복구할 때 필요한 '빈도수 정보'와 '데이터 길이(padding 처리용)'를 저장
        # pickle을 사용해 dictionary를 그대로 바이너리로 저장
        with open(output_path, 'wb') as outfile:
            # 헤더 길이와 헤더 본문 저장
            header = frequency
            pickle.dump(header, outfile)
            
            writer = BitWriter(output_path)
            # 덮어쓰기 모드. 따라서 헤더부터 다시 쓴다.
            writer.file.write(pickle.dumps(header)) # 헤더 기록
        # pickle.load는 알아서 객체 하나만큼만 읽으므로 구분자 불필요.
        
        # 본문(MTF 데이터) 압축
        for num in mtf_data:
            code = self.codes[num]
            for bit in code:
                writer.write_bit(int(bit))
        
        writer.close()
        
        # 결과 출력
        original_size = os.path.getsize(input_path)
        compressed_size = os.path.getsize(output_path)
        ratio = (1 - compressed_size / original_size) * 100
        print(f"압축 완료! ({original_size}B -> {compressed_size}B, 압축률 {ratio:.2f}%)")

# 6. 해제기(Decompressor) 클래스
class HuffmanDecompressor:
    def __init__(self):
        self.root = None

    # 압축기와 동일한 로직으로 트리를 복구
    def build_tree(self, frequency):
        heap = []
        for key, freq in frequency.items():
            node = HuffmanNode(key, freq)
            heapq.heappush(heap, node)
        
        if not heap: return None

        while len(heap) > 1:
            node1 = heapq.heappop(heap)
            node2 = heapq.heappop(heap)
            merged = HuffmanNode(None, node1.freq + node2.freq)
            merged.left = node1
            merged.right = node2
            heapq.heappush(heap, merged)
        
        return heapq.heappop(heap)

    def decompress(self, input_path, output_path):
        print(f"[{input_path}] 해제 시작")

        with open(input_path, 'rb') as file:
            # 6-1. 헤더 읽기 (빈도수 정보)
            try:
                frequency = pickle.load(file)
            except EOFError:
                print("파일이 손상되었거나 비어있습니다.")
                return

            # 6-2. 허프만 트리 재구축
            self.root = self.build_tree(frequency)
            
            # 6-3. 비트 스트림 읽기 및 디코딩 (Huffman Decode)
            # 본문 데이터는 pickle 뒤에 이어져 있다.
            bit_string = ""
            # 파일의 나머지 바이트를 전부 읽어서 비트 문자열로 변환
            raw_content = file.read()
            
        # 바이트 -> 비트 문자열 변환 (예: b'\x80' -> '10000000')
        # 대용량 처리 시 속도를 위해 비트 연산으로 최적화 가능
        bits = []
        for byte in raw_content:
            # f'{byte:08b}'는 숫자를 8자리 2진수 문자열로 바꿈
            bits.append(f'{byte:08b}')
        full_bit_string = "".join(bits)
        
        # 6-4. 트리 탐색하며 MTF 리스트 복원
        decoded_mtf = []
        node = self.root
        
        # 총 데이터 개수(Total Count)를 헤더에서 계산해서, 
        # 패딩(Padding)된 뒤쪽 0비트들을 무시하고 정확히 멈춰야 함
        total_symbols = sum(frequency.values())
        decoded_count = 0
        
        for bit in full_bit_string:
            if bit == '0':
                node = node.left
            else:
                node = node.right
            
            # 리프 노드(글자) 도달
            if node.char is not None:
                decoded_mtf.append(node.char)
                decoded_count += 1
                node = self.root # 루트로 복귀
                
                if decoded_count == total_symbols:
                    break # 모든 데이터를 다 찾았으면 종료 (Padding 무시)

        print("1단계: 허프만 디코딩 완료")

        # 6-5. MTF 역변환
        print("2단계: MTF 역변환")
        decoded_bwt = self.mtf_inverse(decoded_mtf)

        # 6-6. BWT 역변환
        print("3단계: BWT 역변환")
        original_bytes = self.bwt_inverse(decoded_bwt)

        # 6-7. 파일 저장
        with open(output_path, 'wb') as outfile:
            # EOS(-1) 제거 (만약 리스트에 남아있다면)
            final_data = bytes([b for b in original_bytes if b != -1])
            outfile.write(final_data)
            
        print("해제 완료! 원본 복구 성공.")

    # 역변환 시 필요한 함수들
    def mtf_inverse(self, mtf_list):
        alphabet = [-1] + list(range(256))
        result = []
        for idx in mtf_list:
            val = alphabet[idx]
            result.append(val)
            alphabet.pop(idx)
            alphabet.insert(0, val)
        return result

    def bwt_inverse(self, bwt_list):
        #LF Mapping을 이용한 고속 BWT 역변환
        # 1. F (First Column) 생성: BWT 리스트를 정렬하면 됨
        # 하지만 정렬보다 빠른 Counting Sort 방식을 응용해 T 벡터 생성
        
        counts = Counter(bwt_list)
        sorted_chars = sorted(counts.keys())
        
        # accum: 특정 문자가 F열(정렬된 열)에서 시작하는 인덱스
        accum = {}
        idx = 0
        for char in sorted_chars:
            accum[char] = idx
            idx += counts[char]
            
        # 2. LF 매핑 벡터(T) 생성
        # T[i] = bwt_list[i]가 F열의 몇 번째 줄로 가는지 가리킴
        n = len(bwt_list)
        T = [0] * n
        current_counts = {} # 현재까지 등장 횟수
        
        for i, char in enumerate(bwt_list):
            rank = current_counts.get(char, 0)
            current_counts[char] = rank + 1
            # 공식: F에서의 시작위치 + 현재까지의 등장 횟수
            T[i] = accum[char] + rank

        # 3. 역추적 (Backtracking)
        # BWT의 복구는 '뒤에서 앞으로' 진행됨
        # EOS(-1)가 문자열의 끝이므로, F열의 -1 위치에서 시작
        
        # F열에서 -1은 무조건 맨 처음(0번 인덱스)에 있음 (가장 작은 값이므로)
        # BWT의 원리에 따라:
        # L[i]는 F[i]의 바로 '앞' 글자임.
        
        reconstructed = []
        # -1(EOS)부터 시작
        curr_idx = 0  # F[0]은 -1임이 보장됨
        
        for _ in range(n):
            # 현재 위치의 L열 값(이전 글자)을 가져옴
            prev_char = bwt_list[curr_idx]
            reconstructed.append(prev_char)
            # 다음 위치로 점프
            curr_idx = T[curr_idx]
            
        # reconstructed는 EOS(-1) 이전 글자부터 거꾸로 담김
        # 예: banana$ -> L열... -> 복구순서: a, n, a, n, a, b, $
        # 따라서 뒤집어야 함.
        reconstructed.reverse()
        
        # 맨 앞의 -1 제거 (우리가 찾은 시작점은 EOS였으므로 리스트 맨 앞에 -1이 옴)
        # 혹은 로직에 따라 맨 뒤일수도 있는데, 위 로직상 -1이 포함됨.
        return [x for x in reconstructed if x != -1]

# 테스트 및 검증 코드
if __name__ == "__main__":
    # 테스트용 텍스트 파일 생성
    with open("test.txt", "w", encoding="utf-8") as f:
        f.write("banana " * 100) # 반복 패턴 생성
        f.write("한글테스트") # 유니코드 테스트

    # 압축 테스트
    compressor = HuffmanCompressor()
    compressor.compress("test.txt", "test.bin")
    
    # 해제 테스트
    decompressor = HuffmanDecompressor()
    decompressor.decompress("test.bin", "recovered.txt")
    
    # 3. 무결성 검증 (파일 내용 비교)
    with open("test.txt", "rb") as f1, open("recovered.txt", "rb") as f2:
        if f1.read() == f2.read():
            print("\n[성공] 원본 파일과 복구된 파일이 완벽하게 일치합니다")
        else:
            print("\n[실패] 파일 내용이 다릅니다. 코드를 확인하세요.")