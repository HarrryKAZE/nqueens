import streamlit as st
import cv2
import numpy as np
from collections import defaultdict
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

### QUEEN PLACEMENT SOLVER (STACKED ADJACENT TRACKING) ###
def solve(territories, grid_size):
    board = [[0]*grid_size for _ in range(grid_size)]
    used_rows = set()
    used_cols = set()
    global_adjacents = set()
    adj_stack = []

    def is_valid(row, col):
        return (
            row not in used_rows and
            col not in used_cols and
            (row, col) not in global_adjacents
        )

    def mark_adjacent(row, col):
        return {
            (row + dr, col + dc)
            for dr in (-1, 0, 1)
            for dc in (-1, 0, 1)
            if 0 <= row + dr < grid_size and 0 <= col + dc < grid_size
        }

    def place_QUEENs(current_territory=0):
        if current_territory == len(territories):
            return True

        for row, col in territories[current_territory]:
            if is_valid(row, col):
                board[row][col] = 1
                used_rows.add(row)
                used_cols.add(col)
                affected = mark_adjacent(row, col)
                global_adjacents.update(affected)
                adj_stack.append(affected)

                if place_QUEENs(current_territory + 1):
                    return True

                board[row][col] = 0
                used_rows.remove(row)
                used_cols.remove(col)
                global_adjacents.difference_update(adj_stack.pop())

        return False

    return board if place_QUEENs() else None

### TERRITORY DETECTION ###
def identify_territories(image, grid_size):
    h, w = image.shape[:2]
    cell_size = h / grid_size

    grid_colors = np.zeros((grid_size, grid_size, 3), dtype=np.uint8)
    for i in range(grid_size):
        for j in range(grid_size):
            y1, y2 = int(i*cell_size + cell_size*0.2), int((i+1)*cell_size - cell_size*0.2)
            x1, x2 = int(j*cell_size + cell_size*0.2), int((j+1)*cell_size - cell_size*0.2)
            grid_colors[i,j] = np.median(image[y1:y2, x1:x2], axis=(0,1))

    pixels = grid_colors.reshape(-1, 3)
    kmeans = KMeans(n_clusters=grid_size, random_state=0).fit(pixels)
    labels = kmeans.labels_.reshape(grid_size, grid_size)

    territories = []
    visited = np.zeros((grid_size, grid_size), dtype=bool)
    directions = [(1,0), (-1,0), (0,1), (0,-1)]

    def bfs(i, j, label):
        queue = [(i, j)]
        territory = []
        while queue:
            x, y = queue.pop(0)
            if visited[x][y]: continue
            visited[x][y] = True
            territory.append([x, y])
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if 0 <= nx < grid_size and 0 <= ny < grid_size:
                    if not visited[nx][ny] and labels[nx][ny] == label:
                        queue.append((nx, ny))
        return territory

    for i in range(grid_size):
        for j in range(grid_size):
            if not visited[i][j]:
                territory = bfs(i, j, labels[i][j])
                territories.append(territory)

    return territories

### STREAMLIT UI WITH MANUAL GRID SELECTION ###
st.set_page_config(layout="wide")
st.title("🎯 LinkedIn styleN-QUEENS Solver using recursive backtracking algorithm")

grid_size = st.radio("Select grid size:", [7, 8, 9, 10], index=0)

uploaded_file = st.file_uploader("Upload grid image:", type=["png","jpg","jpeg"])

if uploaded_file:
    image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    size = max(image.shape[:2])
    image = cv2.resize(image, (size, size))

    verify_img = image.copy()
    cell_size = size / grid_size
    for i in range(1, grid_size):
        cv2.line(verify_img, (0, int(i*cell_size)), (size, int(i*cell_size)), (255,0,0), 2)
        cv2.line(verify_img, (int(i*cell_size), 0), (int(i*cell_size), size), (255,0,0), 2)

    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Original Image")
    with col2:
        st.image(verify_img, caption=f"{grid_size}x{grid_size} Grid Overlay")

    territories = identify_territories(image, grid_size)
    st.subheader(f"Detected {len(territories)} territories")

    fig, ax = plt.subplots(figsize=(8,8))
    ax.imshow(image)
    colors = plt.cm.get_cmap('tab20', len(territories))
    for idx, t in enumerate(territories):
        for i,j in t:
            rect = plt.Rectangle((j*cell_size, i*cell_size), cell_size, cell_size,
                               fill=False, edgecolor=colors(idx), linewidth=3)
            ax.add_patch(rect)
    st.pyplot(fig)

    for i, territory in enumerate(territories):
        st.write(f"**Territory {i+1}**: {len(territory)} cells - {territory}")

    if st.button("Solve QUEEN Placement"):
        solution = solve(territories, grid_size)
        if solution:
            st.success("Solution found!")
            sol_img = image.copy()
            for i in range(grid_size):
                for j in range(grid_size):
                    if solution[i][j]:
                        center = (int((j+0.5)*cell_size), int((i+0.5)*cell_size))
                        
                        cv2.circle(sol_img, center, int(cell_size/4), (0,0,255), -1)

            col3, col4 = st.columns(2)
            with col3:
                st.image(sol_img, caption="Solution (Red circles = QUEENs)")
            with col4:
                st.subheader("QUEEN Placement Matrix")
                st.write(np.array(solution))
        else:
            st.error("No valid solution found")

st.markdown("""
### How to use:
1. **Select grid size** (8x8 or 9x9)
2. **Upload a clear image** of the grid
3. **Verify grid overlay** matches your image
4. **Review detected territories**
5. **Click Solve** to find QUEEN placements

### Tips for best results:
- Use high-contrast images with clear grid lines
- Ensure territories have distinct colors
- Verify territory detection matches your grid
- For 9x9 grids, ensure exactly 9 territories are detected
""")
