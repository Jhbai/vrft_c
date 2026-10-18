# VRFT Constrained API Server

高並發 **MIMO VRFT Controller Design Server**，支援 **Box-Constrained Solver**，並透過 **HTTP/1.1 Binary Stream Wrapper** 提供 Kubernetes Ingress 相容的 API。

---

## ✨ 功能特色
- **多執行緒工作池**：支援 8 個 worker threads，處理高併發請求。
- **MIMO 傳遞函數模擬**：支援多輸入多輸出離散傳遞函數。
- **穩定反演策略**：`default_stable_inversion` 提供穩定的模型反演。
- **Box-Constrained QP Solver**：`solve_box_qp_cd` 使用座標下降法解決帶界限的二次規劃問題。
- **HTTP API**：
  - `POST /design`：接收二進位協定 payload，回傳最佳化參數。
  - `GET /healthz`：健康檢查端點。

---

## 📦 二進位協定格式
### 請求 (Request)
Header + `(u, y, y_iv, Td, C, L, pbounds)`

- `u`：輸入訊號
- `y`：輸出訊號
- `y_iv`：反演輸出
- `Td`：目標模型
- `C`：控制器基底
- `L`：濾波器
- `pbounds`：參數上下界

### 回應 (Response)
`Status + Optimal p`

- `Status`：設計狀態碼
- `Optimal p`：最佳化控制器參數

---
