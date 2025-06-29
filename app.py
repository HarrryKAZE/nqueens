import streamlit as st
import cv2
import numpy as np
from PIL import Image
from io import BytesIO

# Use your existing functions here: load_and_preprocess(), detect_existing_queens(), etc.



st.title("♛ N-Queens Puzzle Solver (Image Input)")

puzzle_img_file = st.file_uploader("Upload Puzzle Image (8x8 board)", type=["png", "jpg", "jpeg"])
queen_img_file = st.file_uploader("Upload Queen Template Image", type=["png", "jpg", "jpeg"])

if puzzle_img_file and queen_img_file:
    img_bytes = np.asarray(bytearray(puzzle_img_file.read()), dtype=np.uint8)
    puzzle_img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)

    queen_bytes = np.asarray(bytearray(queen_img_file.read()), dtype=np.uint8)
    queen_img = cv2.imdecode(queen_bytes, cv2.IMREAD_UNCHANGED)

    # Call your image processing functions here
    def load_and_preprocess(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150, apertureSize=3)

        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 80, minLineLength=50, maxLineGap=10)
        pts = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            pts.extend([[x1, y1], [x2, y2]])
        pts = np.array(pts)

        rect = cv2.minAreaRect(pts)
        box = cv2.boxPoints(rect).astype(np.int32)

        def order_points(pts_array):
            rect_ordered = np.zeros((4, 2), dtype="float32")
            s = pts_array.sum(axis=1)
            rect_ordered[0] = pts_array[np.argmin(s)]
            rect_ordered[2] = pts_array[np.argmax(s)]
            diff = np.diff(pts_array, axis=1)
            rect_ordered[1] = pts_array[np.argmin(diff)]
            rect_ordered[3] = pts_array[np.argmax(diff)]
            return rect_ordered

        src = order_points(box)
        dst = np.array([[0, 0], [799, 0], [799, 799], [0, 799]], dtype="float32")
        M = cv2.getPerspectiveTransform(src, dst)
        warp = cv2.warpPerspective(img, M, (800, 800))
        return warp
    def detect_existing_queens(board_img, tmpl):
        if tmpl.shape[2] == 4:
            gtmpl = cv2.cvtColor(tmpl[:, :, :3], cv2.COLOR_BGR2GRAY)
        else:
            gtmpl = cv2.cvtColor(tmpl, cv2.COLOR_BGR2GRAY)
        gtmpl = cv2.resize(gtmpl, (80, 80))

        cell = 100
        placed = set()
        gray_board = cv2.cvtColor(board_img, cv2.COLOR_BGR2GRAY)

        for r in range(8):
            for c in range(8):
                crop = gray_board[r*cell:(r+1)*cell, c*cell:(c+1)*cell]
                crop = cv2.resize(crop, (80, 80))
                res = cv2.matchTemplate(crop, gtmpl, cv2.TM_CCOEFF_NORMED)
                _, maxv, _, _ = cv2.minMaxLoc(res)
                if maxv > 0.8:
                    placed.add((r, c))
        return placed
    def solve_n_queens(n, fixed_positions):
        solutions = []
        cols, diag1, diag2 = set(), set(), set()
        board = [-1] * n

        for r, c in fixed_positions:
            board[r] = c
            cols.add(c)
            diag1.add(r + c)
            diag2.add(r - c)

        def backtrack(r):
            if r == n:
                solutions.append(board.copy())
                return True
            if board[r] != -1:
                return backtrack(r + 1)
            for c in range(n):
                if c in cols or (r + c) in diag1 or (r - c) in diag2:
                    continue
                board[r] = c
                cols.add(c)
                diag1.add(r + c)
                diag2.add(r - c)
                if backtrack(r + 1): return True
                board[r] = -1
                cols.remove(c)
                diag1.remove(r + c)
                diag2.remove(r - c)
            return False
        backtrack(0)
        return solutions[0] if solutions else []
    def draw_solution(board_img, solution, queen_img):
        out = board_img.copy()
        icon = cv2.resize(queen_img[:, :, :3], (80, 80))

        for r, c in enumerate(solution):
            x = c * 100 + 10
            y = r * 100 + 10
            out[y:y+80, x:x+80] = icon

        return out




    warped = load_and_preprocess(puzzle_img)
    fixed_queens = detect_existing_queens(warped, queen_img)
    solution = solve_n_queens(8, fixed_queens)
    solved_img = draw_solution(warped, solution, queen_img)

    st.image(cv2.cvtColor(solved_img, cv2.COLOR_BGR2RGB), caption="Solved Puzzle", channels="RGB")

    # Optional: Download button
    img_pil = Image.fromarray(cv2.cvtColor(solved_img, cv2.COLOR_BGR2RGB))
    buf = BytesIO()
    img_pil.save(buf, format="PNG")
    st.download_button("Download Solved Image", data=buf.getvalue(), file_name="solved.png")
