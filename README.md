# Hướng dẫn sử dụng Dataset và Đánh giá Marshmallow

Tài liệu này hướng dẫn chi tiết cách thiết lập, trích xuất dữ liệu, xác thực và chạy thử nghiệm đánh giá postconditions (hậu điều kiện) trên thư viện `marshmallow` bằng SpecMind/EvalPlus.

---

## Quy trình Thực hiện (6 Bước)

### Bước 1: Clone mã nguồn Marshmallow từ GitHub
Đầu tiên, bạn cần tải mã nguồn của thư viện `marshmallow` về thư mục làm việc. Khuyên dùng thư mục `./marshmallow` ở ngay thư mục gốc của dự án để đồng bộ với các cấu hình mặc định.

Mở terminal và chạy lệnh sau từ thư mục gốc của dự án:
```bash
git clone https://github.com/marshmallow-code/marshmallow.git ./marshmallow
```

**Yêu cầu cài đặt môi trường:**
Bạn nên cài đặt các thư viện phụ thuộc của `marshmallow` và các thư viện cần thiết cho việc chạy thử nghiệm:
```bash
pip install marshmallow pytest tqdm openai
```
*(Nếu cần cài đặt đầy đủ các gói phát triển của marshmallow để đảm bảo chạy kiểm thử không lỗi, chạy lệnh: `pip install -r ./marshmallow/requirements-dev.txt`)*

---

### Bước 2: Trích xuất tập dữ liệu (Dataset Extraction)
Chạy file [extract_marshmallow_dataset.py](./extract_marshmallow_dataset.py) từ thư mục gốc để trích xuất các hàm, docstring và các ca kiểm thử (tests) tương ứng thành định dạng JSONL.

**Lệnh thực hiện:**
```bash
python extract_marshmallow_dataset.py --repo-root ./marshmallow --output-dir ./out_marshmallow_docfilter --documented-only
```

**Giải thích các tham số chính:**
*   `--repo-root`: Đường dẫn tới thư mục marshmallow đã clone ở Bước 1 (ví dụ: `./marshmallow`).
*   `--output-dir`: Thư mục lưu trữ kết quả trích xuất (nên đặt là `./out_marshmallow_docfilter`).
*   `--documented-only`: Chỉ trích xuất các hàm/phương thức có docstring chính thức trong tài liệu của marshmallow.
*   `--match-mode`: Chế độ ánh xạ các bài test với các hàm. Mặc định là `trace` (chạy pytest trực tiếp để theo vết các hàm được gọi), hoặc dùng `direct` để phân tích tĩnh mã nguồn một cách bảo thủ.

**Kết quả đầu ra của bước này:**
Trong thư mục `./out_marshmallow_docfilter/` sẽ xuất hiện các file:
*   `marshmallow_problems.jsonl`: Chứa danh sách các bài toán (hàm cần sinh hậu điều kiện).
*   `marshmallow_tests.jsonl`: Chứa danh sách các test cases tương ứng liên kết với từng bài toán.

---

### Bước 3: Xác thực và lọc Test Cases dùng được (Validation)
Một số test cases sau khi trích xuất có thể không khả thi hoặc bị lỗi cú pháp do tham số hóa. Chúng ta sử dụng [validate_marshmallow_tests.py](./SpecMind/evalplus/validate_marshmallow_tests.py) để lọc ra các test cases thực sự chạy được bằng pytest.

**Lệnh thực hiện:**
```bash
cd SpecMind/evalplus
python validate_marshmallow_tests.py --tests-path ../../out_marshmallow_docfilter/marshmallow_tests.jsonl --marshmallow-repo-root ../../marshmallow --filtered-output ../../out_marshmallow_docfilter/marshmallow_tests_valid.jsonl
```

**Giải thích hoạt động:**
*   Script sẽ chạy `pytest --collect-only` trên kho lưu trữ `marshmallow` để thu thập toàn bộ các `nodeid` hợp lệ.
*   Sau đó, nó đối chiếu danh sách test cases đã trích xuất, giữ lại những test cases khớp hoàn toàn và ghi ra file `marshmallow_tests_valid.jsonl`.
*   Các test không hợp lệ sẽ được ghi nhận vào `output/marshmallow/invalid_tests.jsonl` kèm theo lý do lỗi.

---

### Bước 4: Cấu hình tham số đánh giá
Trước khi tiến hành chạy đánh giá hàng loạt, bạn hãy mở file [run_marshmallow_all.py](./SpecMind/evalplus/run_marshmallow_all.py) và điều chỉnh các thông số mặc định (nằm ở đầu file, khoảng dòng 43-56) cho phù hợp với nhu cầu thử nghiệm:

| Biến Cấu Hình | Giá Trị Mặc Định | Ý Nghĩa / Cách Chỉnh Sửa |
| :--- | :--- | :--- |
| **`DEFAULT_MODE`** | `"exploratory"` | Chế độ thử nghiệm. Gồm các lựa chọn: `"single-pass"`, `"greedy"`, hoặc `"exploratory"` (khám phá và tối ưu hóa qua các lượt feedback). |
| **`DEFAULT_MAX_TURNS`** | `12` | Số lượt tương tác (turns) tối đa với mô hình LLM cho mỗi bài toán. |
| **`DEFAULT_MODEL`** | `"meta-llama/llama-4-scout"` | Tên mô hình ngôn ngữ lớn (LLM) sẽ gọi qua OpenRouter. |
| **`DEFAULT_PROBLEMS_PATH`** | `"../../out_marshmallow_docfilter/marshmallow_problems.jsonl"` | Đường dẫn tới danh sách bài toán đã trích xuất ở Bước 2. |
| **`DEFAULT_TESTS_PATH`** | `"../../out_marshmallow_docfilter/marshmallow_tests_valid.jsonl"` | Đường dẫn tới danh sách test cases hợp lệ đã lọc ở Bước 3. |
| **`DEFAULT_REPO_ROOT`** | `"../../marshmallow"` | Đường dẫn gốc tới kho mã nguồn marshmallow. |
| **`DEFAULT_RESUME`** | `False` | Đặt thành `True` nếu bạn muốn chạy tiếp tục (skip các bài đã có kết quả đầu ra trước đó) khi quá trình chạy bị gián đoạn. |
| **`DEFAULT_RUN_POWER_EVAL`** | `False` | Bật/Tắt tính năng chạy đánh giá độ mạnh của điều kiện (mutant testing). |
| **`DEFAULT_PYTEST_BATCH_SIZE`**| `100` | Số lượng test cases chạy song song theo lô để tối ưu thời gian. Khuyên dùng từ `50` đến `100`. |

> [!IMPORTANT]
> **Thiết lập API Key:** Bạn cần cấu hình API Key của OpenRouter hoặc OpenAI để công cụ có thể gọi mô hình LLM. Có hai cách thiết lập chính như sau:
> 
> #### Cách 1: Sử dụng tệp tin `.env` (Khuyên dùng)
> Tạo một tệp tin tên là `.env` đặt tại thư mục `SpecMind/evalplus/.env` (hoặc thư mục cha bên ngoài) với nội dung:
> ```env
> OPENROUTER_API_KEY=your_openrouter_api_key_here
> # Hoặc nếu dùng OpenAI trực tiếp:
> # OPENAI_API_KEY=your_openai_api_key_here
> ```
> *(Script `run_marshmallow_all.py` sẽ tự động quét và tải tệp `.env` này khi chạy).*
> 
> #### Cách 2: Thiết lập qua biến môi trường (Environment Variables)
> Trước khi chạy script, bạn nhập lệnh sau vào terminal:
> *   **Trên Windows (PowerShell):**
>     ```powershell
>     $env:OPENROUTER_API_KEY="your_openrouter_api_key_here"
>     ```
> *   **Trên Windows (Command Prompt - CMD):**
>     ```cmd
>     set OPENROUTER_API_KEY=your_openrouter_api_key_here
>     ```
> *   **Trên Linux / macOS:**
>     ```bash
>     export OPENROUTER_API_KEY="your_openrouter_api_key_here"
>     ```

---

### Bước 5: Chạy Đánh giá Toàn bộ (Evaluation Execution)
Sau khi đã hoàn tất các bước chuẩn bị và cấu hình tham số, tiến hành chạy toàn bộ quy trình đánh giá bằng lệnh:

```bash
cd SpecMind/evalplus
python run_marshmallow_all.py
```
*(Bạn cũng có thể ghi đè các cấu hình bằng các tham số dòng lệnh, ví dụ: `python run_marshmallow_all.py --mode single-pass --max-turns 5`)*

---

### Bước 6: Đọc và Hiểu kết quả Output
Sau khi chạy xong, toàn bộ kết quả thử nghiệm sẽ được lưu trữ tại thư mục đầu ra, ví dụ: `SpecMind/evalplus/output/marshmallow/{mode}/baseonly_mu{max_turns}/`

Các tệp tin kết quả sinh ra gồm có:

#### 1. Tệp tin nhật ký `run.log`
Ghi lại toàn bộ tiến trình và thông tin xuất ra màn hình console trong suốt quá trình chạy. Nếu có lỗi phát sinh giữa chừng, bạn có thể kiểm tra tệp tin này để debug.

#### 2. Các tệp tin chi tiết `postcondition_results_MyDataset_*.json`
Mỗi bài toán (hàm) sẽ có một tệp tin kết quả chi tiết tương ứng. Nội dung bao gồm:
*   Mã nguồn hậu điều kiện (postcondition) do mô hình sinh ra.
*   Lịch sử tương tác của từng lượt (turns), bao gồm cả feedback của các ca kiểm thử bị lỗi.
*   Số lượng token tiêu thụ (`token_usage`).
*   Kết quả đánh giá độ chính xác và độ hoàn thiện.

#### 3. Tệp tin tổng hợp `summary.json` và `summary.csv`
Đây là nơi tổng hợp kết quả của toàn bộ thử nghiệm. Các chỉ số quan trọng cần lưu ý:
*   **`task_count`**: Tổng số bài toán (hàm) marshmallow được đưa vào đánh giá.
*   **`success_count`**: Số lượng bài toán sinh thành công hậu điều kiện (vượt qua toàn bộ các ca kiểm thử cơ bản).
*   **`corr_percent` (Correlation / Accuracy)**: Tỷ lệ chính xác của mô hình, tính bằng `(success_count / task_count) * 100`. Chỉ số này càng cao chứng tỏ khả năng sinh hậu điều kiện đúng của mô hình càng tốt.
*   **`avg_completeness_percent`**: Độ hoàn thiện trung bình của các hậu điều kiện đối với các đột biến mã nguồn (mutants). Điểm càng gần 100% chứng tỏ hậu điều kiện càng chặt chẽ và phòng tránh lỗi tốt.
*   **`avg_efficiency_percent` (Hiệu suất E)**: Được tính bằng `độ hoàn thiện / số lượt submit`. Chỉ số này đánh giá khả năng tối ưu hóa của mô hình (đạt độ bao phủ cao nhưng tốn ít lượt tương tác feedback nhất).
*   **`test_passed` / `test_failed` / `test_total`**: Thống kê số lượng test cases của marshmallow vượt qua (passed) hoặc thất bại (failed) trên các hậu điều kiện được sinh ra.
*   **`total_token_usage`**: Tổng lượng token đã tiêu tốn trong suốt quá trình chạy thử nghiệm (chia ra: `prompt_tokens`, `completion_tokens` và `total_tokens`), giúp bạn ước tính chi phí API.
