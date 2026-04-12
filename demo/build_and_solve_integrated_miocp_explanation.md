# Giải thích chi tiết hàm `_build_and_solve_integrated_miocp(...)`

Tài liệu này giải thích chi tiết hàm [`_build_and_solve_integrated_miocp`](./optimizer.py) trong `demo/optimizer.py`. Đây là hàm trung tâm của `IntegratedMIOCPSolver`, nơi bài toán lập kế hoạch chuyển động được dựng thành một bài toán tối ưu liên hợp giữa:

- lựa chọn đường đi rời rạc trên region graph,
- động học liên tục theo multiple shooting,
- ràng buộc hình học on/off bằng Big-M,
- và chi phí quỹ đạo.

Hàm nằm tại [demo/optimizer.py](/home/dang-khoa/code/motion-planning/GCS-MMS/demo/optimizer.py:981).

## 1. Mục tiêu của hàm

Hàm này xây dựng và giải continuous relaxation của một bài toán MIOCP tích hợp một pha:

- Biến rời rạc kiểu nhị phân:
  - `y_uv`: kích hoạt cạnh `(u, v)` trong graph
  - `p_v`: kích hoạt region `v`
- Biến liên tục:
  - `s_v^-`: trạng thái vào region
  - `s_v^+`: trạng thái ra khỏi region
  - `w_v`: tham số điều khiển trong region
  - `delta_v`: thời gian lưu trong region
  - `z_uv`: trạng thái interface trên cạnh region-region
  - `rho_v`: epigraph cost cục bộ của region

Điểm quan trọng là:

- mô hình vẫn giữ cấu trúc “giống bài toán mixed-integer”,
- nhưng backend đang dùng `IPOPT`,
- nên các biến `y_uv`, `p_v` thực tế được relax về đoạn `[0, 1]`,
- sau đó hàm sẽ trích một đường đi khả dĩ từ nghiệm relax,
- rồi nếu cần thì gọi fixed-path NLP để “polish” nghiệm cho liên tục và vật lý hơn.

## 2. Input và output

### Input

- `start_state: np.ndarray`
  - trạng thái đầu `[px, py, theta]`
- `goal_state: np.ndarray`
  - trạng thái đích `[px, py, theta]`
- `verbose: bool`
  - nếu `True`, in ra số biến, số ràng buộc, warm-start path, backend solver

### Output

Trả về `OptimizationResult`, chứa:

- `success`
- `path`, `path_regions`
- `total_cost`
- `solver_status`
- `constraint_violation`
- `max_integrality_gap`
- `entry_states`, `exit_states`
- `control_params`, `time_durations`
- `interface_points`
- `trajectories`, `mesh_samples`
- `defect_norm`
- `max_connection_gap`

## 3. Ý tưởng tổng thể của flow

Hàm làm lần lượt 8 việc lớn:

1. Lấy thông tin kích thước bài toán và tìm một warm-start path trên graph.
2. Dựng initial guess cho các region nằm trên warm path.
3. Khai báo tất cả biến quyết định CasADi và nhớ vị trí của chúng trong vector nghiệm.
4. Dựng các ràng buộc:
   - network flow,
   - zero-when-inactive,
   - exact dynamics + local cost epigraph,
   - region geometry,
   - interface coupling,
   - source/target coupling.
5. Dựng objective `sum(rho_v)`.
6. Gọi IPOPT để giải continuous relaxation.
7. Phân tích nghiệm relax để trích ra một discrete path.
8. Parse nghiệm thành `OptimizationResult`, tính diagnostics, và nếu cần thì polish bằng fixed-path NLP.

## 4. Bước 1: lấy kích thước và warm-start path

Đoạn đầu:

- `n_x = self.dynamics.n_x`
- `n_w = self.control_param.n_w`
- `n_mesh = self.config.n_mesh_points`
- `edges = list(self.graph.graph.edges())`
- `region_nodes = self.graph.get_region_nodes()`

Ý nghĩa:

- `n_x` là số chiều trạng thái, ở demo unicycle là `3 = [px, py, theta]`.
- `n_w` là số chiều vector điều khiển tham số hóa trong một region.
- `n_mesh` là số điểm sample dùng để kiểm tra safety trong region.
- `edges` là toàn bộ cạnh trong graph.
- `region_nodes` là toàn bộ node region kiểu `R0`, `R1`, ...

Sau đó:

- hàm gọi `self._compute_warm_start_path(start_state, goal_state)`.

Helper này dùng shortest path theo khoảng cách centroid trên graph. Nó không phải nghiệm tối ưu cuối cùng, mà chỉ là:

- một discrete path hợp lý để seed initial guess,
- giúp IPOPT bắt đầu từ trạng thái “có cấu trúc” thay vì random.

Nếu không có đường đi, hàm trả về thất bại ngay:

- `success=False`
- `solver_status="No path exists in region graph"`

## 5. Bước 2: dựng initial guess theo warm path

### 5.1. Warm edges và warm regions

Từ `warm_path`, hàm suy ra:

- `warm_edges`
- `warm_edge_set`
- `warm_region_set`

Các tập này được dùng để:

- set initial guess `y_uv = 1` nếu cạnh thuộc warm path,
- set initial guess `p_v = 1` nếu region thuộc warm path,
- khởi tạo state/control/time của region theo hình học local.

### 5.2. Global heading

`global_heading` là góc từ `start_state` đến `goal_state`.

Nó dùng làm fallback cho các trường hợp:

- không suy ra được hướng rõ ràng từ entry anchor đến exit anchor,
- hoặc interface chưa có direction tốt.

### 5.3. `region_init`

Cho từng region nằm trên warm path, hàm tính:

- `entry_pos`
- `exit_pos`
- `direction = exit_pos - entry_pos`
- `heading`
- `distance`
- `delta_init`

Trong đó:

- `entry_pos` lấy từ `incoming_edge`
- `exit_pos` lấy từ `outgoing_edge`
- cả hai đều dùng helper `_get_edge_anchor(...)`

`_get_edge_anchor(...)` trả về một “điểm đại diện” cho cạnh:

- nếu cạnh là `SOURCE -> Rk`, anchor là `start_state[:2]`
- nếu cạnh là `Rk -> TARGET`, anchor là `goal_state[:2]`
- nếu là region-region edge và có intersection, anchor là centroid của intersection
- nếu không, anchor là midpoint giữa centroid hai region

Từ đó, hàm tạo initial guess:

- `s_minus = [entry_x, entry_y, heading]`
- `s_plus = [exit_x, exit_y, heading]`
- `w = [0.5, 0.0] * n_control_segments`
- `delta = clamp(distance / v_max, delta_min, delta_max)`
- `rho = max(a * delta, 1e-3)`

Ý nghĩa trực giác:

- robot ban đầu được giả sử đi tương đối thẳng,
- vận tốc tiến dương nhỏ,
- tốc độ quay bằng 0,
- thời gian trong region tỷ lệ với khoảng cách cần đi qua region đó.

### 5.4. `interface_init`

Cho mỗi region-region edge, hàm tạo initial guess cho `z_uv`.

Nếu edge thuộc warm path:

- `z_uv` được đặt tại anchor của edge,
- heading lấy từ `global_heading` hoặc từ `s_minus` của region downstream.

Nếu edge không thuộc warm path:

- `z_uv = 0`.

`z_uv` là interface state nằm tại “chỗ giao tiếp” giữa hai region, dùng để nối `s_u^+` với `s_v^-`.

## 6. Bước 3: khai báo decision variables

Hàm tạo các dict:

- `y_vars`
- `p_vars`
- `s_minus_vars`
- `s_plus_vars`
- `w_vars`
- `delta_vars`
- `rho_vars`
- `z_vars`

và một cấu trúc `var_slices` để nhớ:

- mỗi biến được đặt ở đoạn nào trong vector lớn `x`.

Đây là điểm rất quan trọng cho phần parse nghiệm sau solver:

- khi có `x_opt`, hàm cần biết phần nào thuộc `y`, phần nào thuộc `s_minus`, ...

### 6.1. `x_vars`, `lbx`, `ubx`, `x0`

Hàm đang dựng bài toán NLP chuẩn của CasADi:

- `x_vars`: danh sách symbol
- `lbx`: lower bounds cho biến
- `ubx`: upper bounds cho biến
- `x0`: initial guess

### 6.2. Hàm phụ `_expand`

`_expand(values, size)` cho phép truyền:

- scalar rồi broadcast lên vector độ dài `size`,
- hoặc truyền trực tiếp đúng vector độ dài `size`.

Ví dụ:

- `lb = 0.0` cho biến scalar hay vector đều hợp lệ,
- `lb = self.state_big_m` cũng hợp lệ nếu đúng kích thước.

### 6.3. Hàm phụ `_register`

`_register(...)` là helper để chuẩn hóa việc:

1. ghi lại slice của biến trong `var_slices`,
2. append symbol vào `x_vars`,
3. append bound vào `lbx`, `ubx`,
4. append initial guess vào `x0`,
5. lưu cờ `is_discrete` vào `discrete`.

Lưu ý:

- `discrete` chỉ là bookkeeping/debug ở đây,
- IPOPT vẫn giải relaxation liên tục,
- tức là các biến có `is_discrete=True` không hề bị ép nhị phân bởi solver.

### 6.4. Các nhóm biến được khai báo

#### Cạnh `y_uv`

Cho mọi cạnh trong graph:

- tạo `y_uv in [0, 1]`
- init = `1` nếu cạnh thuộc warm path, ngược lại `0`

#### Region `p_v`

Cho mọi region:

- tạo `p_v in [0, 1]`
- init = `1` nếu region thuộc warm path

#### Trạng thái vào/ra `s_v^-`, `s_v^+`

Cho mỗi region:

- tạo hai vector kích thước `n_x`
- bound đối xứng `[-state_big_m, state_big_m]`
- init từ `region_init` nếu region nằm trên warm path, không thì zero

#### Điều khiển `w_v`

Cho mỗi region:

- tạo vector kích thước `n_w`
- bound `[-control_big_m, control_big_m]`
- init từ `region_init` hoặc zero

#### Thời gian `delta_v`

Cho mỗi region:

- tạo scalar
- bound `[0, delta_max]`
- init từ warm guess hoặc `0`

Lưu ý:

- cận dưới ở đây là `0`, không phải `delta_min`,
- vì khi region không active thì `delta_v` phải về `0`,
- điều kiện `delta_min` sẽ được áp qua ràng buộc on/off phía sau.

#### Epigraph `rho_v`

Cho mỗi region:

- tạo scalar
- bound `[0, rho_big_m]`
- init từ warm guess hoặc `0`

#### Interface state `z_uv`

Cho mỗi region-region edge:

- tạo vector kích thước `n_x`
- bound đối xứng `[-state_big_m, state_big_m]`
- init từ `interface_init`

## 7. Bước 4: gom toàn bộ biến thành vector `x`

Sau khi đăng ký xong toàn bộ biến:

- `x = ca.vertcat(*x_vars)`

Đây là decision vector hoàn chỉnh của bài toán.

Từ thời điểm này:

- toàn bộ bài toán được biểu diễn dưới dạng:
  - `min f(x)`
  - sao cho `lbx <= x <= ubx`
  - và `lbg <= g(x) <= ubg`

## 8. Bước 5: dựng danh sách ràng buộc

Hàm tạo:

- `g`: danh sách biểu thức constraint
- `lbg`: lower bound cho từng phần tử của `g`
- `ubg`: upper bound cho từng phần tử của `g`

Hai helper:

- `add_eq(expr)` tương đương ép `expr == 0`
- `add_leq(expr)` tương đương ép `expr <= 0`

### 8.1. Layer 1: network flow constraints

Đây là lớp ràng buộc rời rạc trên graph.

#### Source flow

```text
sum_{e in source_edges} y_e = 1
```

Ý nghĩa:

- từ `SOURCE`, tổng lưu lượng đi ra phải bằng 1.

#### Target flow

```text
sum_{e in target_edges} y_e = 1
```

Ý nghĩa:

- tổng lưu lượng đi vào `TARGET` phải bằng 1.

#### Flow conservation tại mỗi region

Với mỗi region `v`:

```text
sum_{u -> v} y_uv = p_v
sum_{v -> w} y_vw = p_v
```

Ý nghĩa:

- nếu region được active (`p_v = 1`) thì:
  - đúng 1 flow đi vào,
  - đúng 1 flow đi ra.
- nếu region không active (`p_v = 0`) thì:
  - không flow nào được đi vào/ra.

Trong continuous relaxation:

- `p_v`, `y_uv` có thể fractional,
- nên constraint này đang tạo một flow liên tục thay vì path nhị phân thật sự.

### 8.2. Layer 2a: zero-when-inactive constraints

Mục tiêu của khối này là:

- nếu `p_v = 0`, toàn bộ biến của region phải “tắt”.

Cho từng region:

#### State on/off

```text
s_v^- <= M_state * p_v
-s_v^- <= M_state * p_v
s_v^+ <= M_state * p_v
-s_v^+ <= M_state * p_v
```

Khi `p_v = 0`:

- suy ra `s_v^- = 0`, `s_v^+ = 0`.

Khi `p_v = 1`:

- state chỉ cần nằm trong hộp Big-M, chưa bị ép bằng gì thêm.

#### Control on/off

Tương tự cho `w_v`.

#### Time on/off

```text
delta_v <= delta_max * p_v
delta_min * p_v <= delta_v
```

Suy ra:

- nếu `p_v = 0` thì `delta_v = 0`,
- nếu `p_v = 1` thì `delta_v in [delta_min, delta_max]`.

Đây là lý do upper/lower bound của `delta_v` ở phần variable registration chỉ cần `[0, delta_max]`.

#### Cost epigraph on/off

```text
rho_v <= rho_big_m * p_v
```

Khi `p_v = 0`:

- `rho_v = 0` do thêm bound `rho_v >= 0`.

### 8.3. Layer 4: exact defect dynamics và local cost epigraph

Đây là phần động học continuous, giữ nguyên cho mọi region.

#### Defect constraint

```text
s_v^+ = F_endpoint(s_v^-, w_v, delta_v)
```

Trong code:

```python
add_eq(s_plus - self.F_endpoint(s_minus, w, delta))
```

Ý nghĩa:

- trạng thái ra khỏi region phải đúng bằng kết quả tích phân động học bắt đầu từ trạng thái vào, dưới control `w_v`, trong thời gian `delta_v`.

Đây là multiple-shooting defect constraint.

#### Local cost epigraph

`local_cost_fn(...)` trả về chi phí cục bộ của region, thường dạng:

```text
a * delta_v + integral( w_L * ||q_dot||^2 + w_E * ||u||^2 )
```

Hàm không cộng trực tiếp local cost vào objective, mà dùng epigraph:

```text
local_cost_v <= rho_v
```

và objective cuối cùng là:

```text
min sum_v rho_v
```

Lợi ích của cách viết này:

- chuẩn hóa objective theo từng region,
- dễ gắn on/off với `rho_v`,
- và hợp với form integrated formulation đang dùng.

### 8.4. Layer 2b: on/off region geometry constraints

Cho từng region `v`, cần đảm bảo quỹ đạo active nằm trong convex set của region.

Region được mô tả ở dạng H-polytope:

```text
Q_v = { x | A_v x <= b_v }
```

#### Endpoint closure

Cho `s_v^-[:2]` và `s_v^+[:2]`, hàm dùng:

```text
A_v p - b_v - boundary_tolerance <= M_position * (1 - p_v)
```

Trong code:

```python
closure_violation = A_dm @ endpoint - b_dm - boundary_tol
add_leq(closure_violation - self.position_big_m * (1 - p))
```

Ý nghĩa:

- nếu `p_v = 1`, endpoint phải nằm trong closure của region,
- được phép chạm biên,
- điều này quan trọng khi hai region chỉ tiếp xúc trên boundary.

#### Interior mesh points

Hàm dùng:

```python
mesh_positions = self.mesh_sampler(s_minus, w, delta)
```

để lấy các điểm sample dọc quỹ đạo trong region.

Nhưng không kiểm tra tất cả mesh points:

- chỉ kiểm tra các điểm interior qua `_interior_mesh_indices(n_mesh)`,
- bỏ qua hai đầu mút.

Ràng buộc:

```text
A_v p_k - b_v + safety_margin <= M_position * (1 - p_v)
```

Ý nghĩa:

- nếu region active, các điểm interior phải nằm strict inside region với margin dương,
- tránh quỹ đạo lướt sát biên hoặc vô tình chạm obstacle do sai số số học.

Tóm lại:

- endpoint: được chạm biên,
- interior mesh points: phải nằm sâu hơn vào trong.

### 8.5. Layer 3: interface membership và coupling cho region-region edges

Đây là lớp ràng buộc nối hai region liên tiếp qua cạnh `u -> v`.

#### On/off của `z_uv`

```text
z_uv <= M_state * y_uv
-z_uv <= M_state * y_uv
```

Khi `y_uv = 0`:

- `z_uv = 0`.

Khi `y_uv = 1`:

- `z_uv` được tự do trong hộp Big-M.

#### `z_uv` phải thuộc cả hai region

Với `z_pos = z_uv[:2]`, hàm áp:

```text
A_u z_pos <= b_u + boundary_tolerance + M_position * (1 - y_uv)
A_v z_pos <= b_v + boundary_tolerance + M_position * (1 - y_uv)
```

Ý nghĩa:

- nếu cạnh active, interface point phải nằm trong closure của cả region nguồn lẫn region đích,
- nên nó là một điểm hợp lệ để “chuyển vùng”.

#### Matching với đầu ra/đầu vào region

```text
s_u^+ = z_uv
s_v^- = z_uv
```

được viết dưới dạng Big-M:

```text
s_u^+ - z_uv <= M_interface * (1 - y_uv)
-(s_u^+ - z_uv) <= M_interface * (1 - y_uv)
s_v^- - z_uv <= M_interface * (1 - y_uv)
-(s_v^- - z_uv) <= M_interface * (1 - y_uv)
```

Ý nghĩa:

- nếu cạnh active, trạng thái cuối region `u` phải trùng với interface state,
- trạng thái đầu region `v` cũng phải trùng với interface state,
- nhờ vậy quỹ đạo liên tục tại nơi nối region.

### 8.6. Source và target boundary coupling

Đây là cách gắn discrete path với điểm đầu/cuối thật.

#### Source edges

Cho mọi cạnh `SOURCE -> v`:

```text
s_v^-[:2] = start_state[:2]  nếu y_SOURCE,v = 1
```

được viết bằng Big-M.

#### Target edges

Cho mọi cạnh `u -> TARGET`:

```text
s_u^+[:2] = goal_state[:2]  nếu y_u,TARGET = 1
```

được viết bằng Big-M.

Lưu ý:

- heading đầu/cuối không bị ép bằng `start_state[2]`, `goal_state[2]`,
- chỉ vị trí được match.
- điều này phù hợp với phần `PathNLPSolver` trong file: robot có thể tự xoay và đến goal ở bất kỳ góc nào.

## 9. Bước 6: objective

Objective được đặt là:

```python
cost = ca.sum1(ca.vertcat(*[rho_vars[node_id] for node_id in region_nodes]))
```

tức:

```text
min sum_v rho_v
```

Do đã có `local_cost_v <= rho_v`, nghiệm tối ưu sẽ đẩy `rho_v` xuống sát local cost thật.

Hiểu ngắn gọn:

- `rho_v` là cost đại diện của region,
- tổng `rho_v` là tổng chi phí toàn quỹ đạo.

## 10. Bước 7: dựng NLP CasADi và gọi IPOPT

Hàm gom:

- `x`
- `f = cost`
- `g = g_expr`

thành:

```python
nlp = {'x': x, 'f': cost, 'g': g_expr}
```

rồi tạo solver:

```python
solver = ca.nlpsol('integrated_miocp_relaxed', 'ipopt', nlp, opts)
```

Options chính:

- `ipopt.max_iter`
- `ipopt.tol`
- `ipopt.print_level`
- `print_time = 0`

Nếu `verbose=True`, hàm in:

- số lượng biến,
- số biến “binary-like” theo bookkeeping,
- số ràng buộc,
- warm-start path,
- backend solver.

Sau đó solver được gọi với:

- `x0`
- `lbx`, `ubx`
- `lbg`, `ubg`

## 11. Bước 8: đọc nghiệm relax

Sau khi solve:

- `x_opt` là vector nghiệm,
- `g_val` là giá trị constraint tại nghiệm,
- `stats` là thống kê solver,
- `success` là cờ thành công theo IPOPT.

### 11.1. Đọc `y` và `p`

Hàm dùng `var_slices` để đọc:

- `edge_values[edge] = y_uv`
- `region_values[node] = p_v`

### 11.2. Tính integrality gap

`_compute_integrality_gap(values)` tính:

```text
max_i min(value_i, 1 - value_i)
```

Ý nghĩa:

- nếu tất cả biến gần 0 hoặc 1 thì gap nhỏ,
- nếu có biến gần 0.5 thì gap lớn.

Đây là cách đo mức độ “fractional” của nghiệm relax.

### 11.3. Trích discrete path

Hàm gọi:

```python
path = self._extract_active_path(
    edge_values,
    prefer_weighted=max_integrality_gap > 1e-3
)
```

Helper này có hai mode:

#### Mode 1: greedy

Nếu integrality gap nhỏ:

- bắt đầu từ `SOURCE`,
- mỗi bước chọn outgoing edge có `y > 0.5`,
- nếu không có thì chọn edge có `y > 1e-3`,
- mỗi bước lấy edge có giá trị lớn nhất.

#### Mode 2: weighted shortest path

Nếu nghiệm fractional đáng kể:

- dựng graph phụ chỉ gồm các edge có weight dương,
- gán trọng số `-log(y_uv)`,
- rồi chạy shortest path.

Ý tưởng:

- edge có `y_uv` lớn thì `-log(y_uv)` nhỏ,
- nên đường đi thu được ưu tiên các edge “được solver chọn mạnh”.

Nếu không trích được path:

- hàm trả thất bại,
- nhưng vẫn trả lại `total_cost`, `constraint_violation`, và solver status.

## 12. Bước 9: dựng `OptimizationResult` từ path đã trích

Khi đã có `path`, hàm:

- chuyển path thành `path_regions`
- tạo `OptimizationResult`
- lưu:
  - `success`
  - `path`
  - `path_regions`
  - `total_cost`
  - `solver_status`
  - `constraint_violation`
  - `max_integrality_gap`

Lưu ý:

- `success` ở đây là success của IPOPT trên bài toán relaxation,
- không có nghĩa discrete path chắc chắn đã “sạch”.

## 13. Bước 10: giải mã nghiệm continuous theo path

Hàm tạo một `RK4Integrator` để rebuild trajectory phục vụ visualization và diagnostics.

Cho từng region trong path:

- đọc `s_minus`, `s_plus`, `w`, `delta` từ `x_opt`
- lưu vào:
  - `entry_states`
  - `exit_states`
  - `control_params`
  - `time_durations`
- gọi `integrate_with_trajectory(...)` để tạo trajectory sample
- gọi `mesh_sampler(...)` để lưu `mesh_samples`

Đây là bước biến nghiệm vector thuần số thành cấu trúc có ý nghĩa hình học và động học.

## 14. Bước 11: tính defect norm và interface points

Với từng bước chuyển trên path:

- hàm tính lại:

```text
defect = || s_v^+ - Integrate(s_v^-, w_v, delta_v) ||
```

- rồi lấy `max(defect_norms)`.

Ý nghĩa:

- dù defect constraint đã được enforce trong NLP,
- vẫn nên tính lại bằng code số để có một chỉ số chẩn đoán dễ đọc.

Đồng thời:

- nếu cạnh kế tiếp có `z` trong `var_slices`, interface point được lấy từ nghiệm `z_uv[:2]`,
- nếu không thì fallback sang `s_plus[:2]`.

Danh sách `interface_points` này chủ yếu để vẽ visualization.

## 15. Bước 12: tính connection gap

Hàm gọi `_compute_connection_gap(...)` để đo:

- khoảng cách từ start đến `entry` region đầu,
- khoảng cách giữa `exit` region trước và `entry` region sau,
- khoảng cách từ `exit` region cuối đến goal.

Đây là metric hình học quan trọng:

- nếu `max_connection_gap` lớn,
- discrete path có thể đúng nhưng trajectory chưa thực sự continuous sau relaxation.

## 16. Bước 13: khi nào thì polish?

Nếu:

- `result.success == True`
- và một trong hai điều kiện sau xảy ra:
  - `max_integrality_gap > 1e-3`
  - `max_connection_gap > 1e-4`

thì hàm gọi:

```python
return self._polish_relaxed_path(path, start_state, goal_state, result)
```

### Ý nghĩa của polish

`_polish_relaxed_path(...)`:

1. lấy path đã trích,
2. lấy continuous variables của relaxation làm warm-start,
3. gọi `PathNLPSolver.solve_path(...)` để giải lại bài toán continuous với path cố định.

Khi path đã cố định:

- không còn biến `y`, `p`,
- không còn ambiguity do relaxation,
- continuity được enforce trực tiếp giữa các region liên tiếp.

Nếu polish thất bại:

- hàm thử thêm vài candidate path gần đó qua `_fallback_fixed_path_search(...)`.

Nếu fallback vẫn thất bại:

- trả lại `relaxation_result`,
- nhưng set `success = False`.

## 17. Ý nghĩa toán học của toàn hàm

Nếu viết gọn, hàm đang giải bài toán kiểu:

```text
minimize    sum_v rho_v

subject to
    flow constraints on graph
    on/off activation constraints
    multiple-shooting defect constraints
    region membership constraints
    interface consistency constraints
    start/goal coupling constraints
    local_cost_v <= rho_v
```

Trong đó:

- discrete structure được giữ lại qua `y_uv`, `p_v`,
- nhưng được relax về liên tục,
- nên đây không phải MILP/MINLP exact solve,
- mà là continuous relaxation của integrated MIOCP.

## 18. Tại sao thiết kế này hợp lý

Hàm này có 3 điểm mạnh rõ ràng:

### 18.1. Discrete và continuous được solve cùng lúc

Thay vì:

1. enumerate path,
2. solve continuous NLP cho từng path,

hàm solve tất cả trong một mô hình thống nhất.

Điều này:

- giảm nhu cầu duyệt path quá nhiều,
- cho phép continuous cost tác động ngược lên lựa chọn region/edge.

### 18.2. Vẫn giữ được interpretability

Mặc dù là integrated relaxation, code vẫn tách constraint thành các lớp rõ ràng:

- flow,
- on/off,
- dynamics,
- geometry,
- interface.

Việc này làm mô hình dễ debug và dễ giải thích.

### 18.3. Có cơ chế “hậu xử lý” thực dụng

Relaxation có thể fractional hoặc hở mối nối.

Thay vì đòi nghiệm relax phải hoàn hảo, code chấp nhận:

- relax để tìm discrete structure tốt,
- rồi polish bằng fixed-path NLP để lấy quỹ đạo usable.

Đây là một chiến lược rất thực dụng.

## 19. Những chi tiết dễ bỏ sót nhưng quan trọng

### 19.1. `discrete` không làm IPOPT thành mixed-integer solver

List `discrete` chỉ được dùng để đếm số biến “binary-like” trong log:

```python
print(f"Integrated relaxation variables: {x.numel()} ({sum(discrete)} relaxed binary-like)")
```

Nó không được truyền vào IPOPT để ép integrality.

### 19.2. `delta_v` có bound dưới bằng 0 ở phần variable, nhưng không mâu thuẫn

Điều này là có chủ đích:

- region inactive cần `delta_v = 0`,
- region active thì nhờ constraint `delta_min * p_v <= delta_v` nên `delta_v >= delta_min`.

### 19.3. Geometry dùng hai mức “độ nghiêm”

- endpoint: closure, cho chạm biên
- interior mesh: strict with safety margin

Đây là cách cân bằng giữa:

- tính khả thi khi region chỉ chạm nhau ở biên,
- và tính an toàn khi quỹ đạo đi bên trong.

### 19.4. Path extraction là heuristic hậu xử lý

Sau relaxation, path không phải lúc nào cũng rơi ra trực tiếp.

`_extract_active_path(...)` là bước heuristic nhưng hợp lý:

- greedy khi nghiệm gần nhị phân,
- weighted shortest path khi nghiệm còn fractional.

### 19.5. `success=True` của IPOPT chưa chắc là nghiệm cuối cùng tốt

Vì bài toán đang là relaxation:

- IPOPT có thể hội tụ tốt,
- nhưng discrete path vẫn fractional,
- hoặc trajectory sau parse vẫn có gap.

Do đó hàm còn cần:

- `max_integrality_gap`
- `max_connection_gap`
- và bước `polish`.

## 20. Pseudocode ngắn gọn

```text
find warm path
if no path:
    return failure

build warm initial guesses for regions and interfaces
declare all variables y, p, s-, s+, w, delta, rho, z
build stacked NLP variable x

add network-flow constraints
add zero-when-inactive constraints
add exact dynamics and local-cost epigraph constraints
add on/off region geometry constraints
add interface membership and coupling constraints
add source/target coupling constraints

objective = sum rho
solve NLP relaxation with IPOPT

extract y and p values
compute integrality gap
extract a path from edge weights
if path extraction fails:
    return failure result

parse continuous states/controls/times on extracted path
reconstruct trajectory and mesh samples
compute defect norm and connection gap

if solution successful but still fractional/discontinuous:
    polish with fixed-path NLP

return result
```

## 21. Liên hệ với các hàm phụ trong file

Để hiểu trọn vẹn hàm này, nên đọc cùng:

- [demo/optimizer.py](/home/dang-khoa/code/motion-planning/GCS-MMS/demo/optimizer.py:737)
  - `_get_edge_anchor(...)`
- [demo/optimizer.py](/home/dang-khoa/code/motion-planning/GCS-MMS/demo/optimizer.py:743)
  - `_compute_warm_start_path(...)`
- [demo/optimizer.py](/home/dang-khoa/code/motion-planning/GCS-MMS/demo/optimizer.py:841)
  - `_compute_integrality_gap(...)`
- [demo/optimizer.py](/home/dang-khoa/code/motion-planning/GCS-MMS/demo/optimizer.py:847)
  - `_extract_active_path(...)`
- [demo/optimizer.py](/home/dang-khoa/code/motion-planning/GCS-MMS/demo/optimizer.py:901)
  - `_polish_relaxed_path(...)`
- [demo/optimizer.py](/home/dang-khoa/code/motion-planning/GCS-MMS/demo/optimizer.py:97)
  - `_compute_connection_gap(...)`
- [demo/optimizer.py](/home/dang-khoa/code/motion-planning/GCS-MMS/demo/optimizer.py:140)
  - `_interior_mesh_indices(...)`

## 22. Kết luận ngắn

`_build_and_solve_integrated_miocp(...)` là hàm biến toàn bộ bài toán motion planning thành một NLP relaxation lớn, trong đó:

- graph flow chọn path,
- Big-M bật/tắt region và interface,
- multiple shooting mô hình hóa động học chính xác,
- local epigraph cost gom objective theo region,
- và phần hậu xử lý chuyển nghiệm relax thành một trajectory dùng được.

Nếu nhìn theo vai trò hệ thống, đây là “bộ máy tích hợp” nối:

- graph search,
- convex geometry,
- dynamics transcription,
- nonlinear programming,
- và trajectory reconstruction

vào cùng một pipeline.
