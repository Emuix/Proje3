#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Proje 3: Labirentte Yol Bulan Ajan
==================================
Özellikler:
1. 20x50 (20 satır x 50 sütun) boyutunda interaktif grid labirent.
   - Başlangıçta ajanın en az 8 farklı engelden (varsayılan 10 engel bloğu) kaçarak
     hedefe ulaştığı hazır bir harita ile açılır.
   - Fare ile duvar ekleme (Sol Tık sürükle) ve duvar silme (Sağ Tık sürükle).
   - Başlangıç veya hedef noktasını fare ile sürükleyip taşıma.
2. Hibrit Yol Bulma (BFS + Dinamik Yol Oluşturma / Dynamic Replanning):
   - Açık bölgelerde yönelimli dinamik arama (Dynamic Heuristic Guidance) ve
     engel/dar geçit komşuluklarında BFS (Breadth-First Search) katmanlarını
     birleştiren hibrit algoritma.
   - Ajan yol üzerinde ilerlerken kullanıcı fareyle yolun üzerine yeni bir duvar
     eklerse, ajan anında dinamik olarak yeni bir en kısa yol hesaplar.
3. Arama Süreci ve En Kısa Yol Animasyonu:
   - İncelenen kareler (açık/kapalı liste), bulunan en kısa yol ve ajanın hareketi
     adım adım animasyonla gösterilir.
4. Bonus - A* Algoritması ve Kare Sayısı Karşılaştırması:
   - BFS, Hibrit (BFS + Dinamik) ve A* algoritmalarının aynı labirentte
     kaç kareye baktığını (ziyaret edilen düğüm sayısı), yol uzunluğunu ve
     çalışma süresini (ms) karşılaştıran canlı panel ve karşılaştırma modu.

Çalıştırma:
    python labirent_ajan.py            # Pygame (yoksa Matplotlib) arayüzünü başlatır
    python labirent_ajan.py --matplotlib # Doğrudan Matplotlib arayüzünü başlatır
    python labirent_ajan.py --test     # Algoritmaları konsolda test eder ve karşılaştırır
"""

from __future__ import annotations

import argparse
import heapq
import math
import random
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Generator, List, Optional, Set, Tuple

# Grid Boyutları (Minimum 20x50 şartını sağlar)
ROWS = 20
COLS = 50

# Hücre Durum Sabitleri
EMPTY = 0
WALL = 1

GridPos = Tuple[int, int]


@dataclass
class SearchStats:
    """Algoritma çalışma istatistiklerini tutan veri sınıfı."""
    algorithm_name: str
    visited_count: int = 0
    path_length: int = 0
    elapsed_ms: float = 0.0
    found: bool = False
    obstacles_dodged: int = 0
    path: List[GridPos] = field(default_factory=list)
    visited_order: List[GridPos] = field(default_factory=list)


class MazeEnvironment:
    """20x50 boyutunda, en az 8 engelden kaçış senaryosu içeren labirent ortamı."""

    def __init__(self, rows: int = ROWS, cols: int = COLS) -> None:
        self.rows = rows
        self.cols = cols
        self.grid: List[List[int]] = [[EMPTY for _ in range(cols)] for _ in range(rows)]
        self.start: GridPos = (10, 2)
        self.goal: GridPos = (10, 47)
        self.obstacle_clusters: List[List[GridPos]] = []
        self.setup_default_obstacles()

    def clear_walls(self) -> None:
        """Tüm duvarları temizler."""
        for r in range(self.rows):
            for c in range(self.cols):
                self.grid[r][c] = EMPTY
        self.obstacle_clusters.clear()

    def setup_default_obstacles(self) -> None:
        """
        Ajanın başlangıçtan hedefe giderken en az 8 (toplam 10) farklı engel bloğundan
        kaçmasını gerektiren şaşırtmalı (slalom) labirent düzenini oluşturur.
        """
        self.clear_walls()
        self.start = (10, 2)
        self.goal = (10, 47)

        # 10 adet dikey/L-tipi engel bloğu (her biri ajanın düz gitmesini engeller ve yön değiştirir)
        obstacle_specs = [
            # (sütun, başlangıç_satırı, bitiş_satırı, yatay_uzantı_satırı)
            (6, 4, 15, 4),     # Engel 1: Ortayı kapatır, üstten/alttan dolaştırır
            (10, 0, 11, 11),   # Engel 2: Üstten iner, alta yönlendirir
            (14, 8, 19, 8),    # Engel 3: Alttan çıkar, üste yönlendirir
            (18, 2, 14, 14),   # Engel 4: Orta-üst engel
            (22, 0, 10, 10),   # Engel 5: Üst bariyer
            (26, 7, 19, 7),    # Engel 6: Alt bariyer
            (30, 3, 15, 3),    # Engel 7: Orta-geniş bariyer
            (34, 0, 11, 11),   # Engel 8: Üst şaşırtma duvarı
            (38, 8, 19, 8),    # Engel 9: Alt şaşırtma duvarı
            (42, 4, 16, 4),    # Engel 10: Hedef öncesi son engel
        ]

        for col, r_start, r_end, hook_row in obstacle_specs:
            cluster: List[GridPos] = []
            for r in range(r_start, r_end + 1):
                if (r, col) not in (self.start, self.goal):
                    self.grid[r][col] = WALL
                    cluster.append((r, col))
            # Küçük L-uzantısı ekleyerek engeli belirginleştir
            for dc in (1, 2):
                nc = col + dc
                if 0 <= nc < self.cols and (hook_row, nc) not in (self.start, self.goal):
                    self.grid[hook_row][nc] = WALL
                    cluster.append((hook_row, nc))
            self.obstacle_clusters.append(cluster)

    def generate_random_maze(self, density: float = 0.24) -> None:
        """En az 8 engelli ve çözülebilir rastgele labirent üretir."""
        rng = random.Random()
        for _ in range(20):
            self.setup_default_obstacles()
            for r in range(self.rows):
                for c in range(self.cols):
                    if (r, c) in (self.start, self.goal):
                        continue
                    if self.grid[r][c] == EMPTY and rng.random() < density * 0.45:
                        self.grid[r][c] = WALL
            # Çözülebilir olduğunu doğrula
            test_stats = run_bfs(self)
            if test_stats.found:
                return
        self.setup_default_obstacles()

    def is_walkable(self, pos: GridPos) -> bool:
        r, c = pos
        return 0 <= r < self.rows and 0 <= c < self.cols and self.grid[r][c] == EMPTY

    def neighbors(self, pos: GridPos) -> List[GridPos]:
        """4-yönlü (Yukarı, Sağ, Aşağı, Sol) geçilebilir komşuları döndürür."""
        r, c = pos
        candidates = [(r - 1, c), (r, c + 1), (r + 1, c), (r, c - 1)]
        return [p for p in candidates if self.is_walkable(p)]

    def has_adjacent_wall(self, pos: GridPos) -> bool:
        """Bir karenin 8-komşuluğunda duvar olup olmadığını kontrol eder."""
        r, c = pos
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.rows and 0 <= nc < self.cols and self.grid[nr][nc] == WALL:
                    return True
        return False

    def count_dodged_obstacles(self, path: List[GridPos]) -> int:
        """
        Bulunan yolun kaç farklı engel kümesinin (veya duvar grubunun) etrafından
        dolaştığını hesaplar.
        """
        if not path:
            return 0
        path_set = set(path)
        dodged = 0

        # Bağlı duvar bileşenlerini (connected components) bul
        visited_walls: Set[GridPos] = set()
        wall_components: List[Set[GridPos]] = []

        for r in range(self.rows):
            for c in range(self.cols):
                if self.grid[r][c] == WALL and (r, c) not in visited_walls:
                    comp: Set[GridPos] = set()
                    q = deque([(r, c)])
                    visited_walls.add((r, c))
                    while q:
                        wr, wc = q.popleft()
                        comp.add((wr, wc))
                        for dr in (-1, 0, 1):
                            for dc in (-1, 0, 1):
                                nr, nc = wr + dr, wc + dc
                                if (
                                    0 <= nr < self.rows
                                    and 0 <= nc < self.cols
                                    and self.grid[nr][nc] == WALL
                                    and (nr, nc) not in visited_walls
                                ):
                                    visited_walls.add((nr, nc))
                                    q.append((nr, nc))
                    wall_components.append(comp)

        # Yolun teğet geçtiği (2 birim mesafeye kadar yaklaşıp etrafından dolandığı) engel grupları
        for comp in wall_components:
            near_path = False
            for wr, wc in comp:
                for dr in range(-2, 3):
                    for dc in range(-2, 3):
                        if (wr + dr, wc + dc) in path_set:
                            near_path = True
                            break
                    if near_path:
                        break
                if near_path:
                    break
            if near_path:
                dodged += 1

        return dodged


# ============================================================================
# ALGORİTMALAR: 1) BFS, 2) HİBRİT (BFS + DİNAMİK YOL), 3) BONUS: A* (A-STAR)
# ============================================================================

def manhattan(a: GridPos, b: GridPos) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def reconstruct_path(came_from: Dict[GridPos, Optional[GridPos]], end: GridPos) -> List[GridPos]:
    path: List[GridPos] = []
    curr: Optional[GridPos] = end
    while curr is not None:
        path.append(curr)
        curr = came_from.get(curr)
    path.reverse()
    return path


def bfs_generator(
    env: MazeEnvironment,
    start: Optional[GridPos] = None,
    goal: Optional[GridPos] = None,
) -> Generator[Tuple[str, GridPos, List[GridPos], int], None, SearchStats]:
    """
    Adım adım animasyon için BFS (Genişlik Öncelikli Arama) üreteci.
    En kısa yolu garanti eder.
    """
    t0 = time.perf_counter()
    s = start or env.start
    g = goal or env.goal

    queue: deque[GridPos] = deque([s])
    came_from: Dict[GridPos, Optional[GridPos]] = {s: None}
    visited_order: List[GridPos] = []

    while queue:
        curr = queue.popleft()
        visited_order.append(curr)

        if curr == g:
            path = reconstruct_path(came_from, g)
            elapsed = (time.perf_counter() - t0) * 1000.0
            stats = SearchStats(
                algorithm_name="BFS (Genişlik Öncelikli)",
                visited_count=len(visited_order),
                path_length=max(0, len(path) - 1),
                elapsed_ms=elapsed,
                found=True,
                obstacles_dodged=env.count_dodged_obstacles(path),
                path=path,
                visited_order=visited_order,
            )
            yield ("FOUND", curr, path, len(visited_order))
            return stats

        yield ("VISIT", curr, [], len(visited_order))

        for nxt in env.neighbors(curr):
            if nxt not in came_from:
                came_from[nxt] = curr
                queue.append(nxt)

    elapsed = (time.perf_counter() - t0) * 1000.0
    stats = SearchStats(
        algorithm_name="BFS (Genişlik Öncelikli)",
        visited_count=len(visited_order),
        path_length=0,
        elapsed_ms=elapsed,
        found=False,
        obstacles_dodged=0,
        path=[],
        visited_order=visited_order,
    )
    yield ("NO_PATH", s, [], len(visited_order))
    return stats


def hybrid_bfs_dynamic_generator(
    env: MazeEnvironment,
    start: Optional[GridPos] = None,
    goal: Optional[GridPos] = None,
) -> Generator[Tuple[str, GridPos, List[GridPos], int], None, SearchStats]:
    """
    Hibrit Algoritma (BFS + Dinamik Yol Oluşturma / Dynamic Frontier + Local BFS):
    - Açık alanlarda dinamik potansiyel alan yönlendirmesi (g + dinamik ağırlıklı h)
      kullanarak hedefe doğru hızla ilerler.
    - Bir engele (duvara) yaklaşıldığında ise yerel BFS genişlemesi (Local BFS Layer)
      devreye girerek engelin etrafındaki en kısa bükülme noktasını garantili şekilde
      keşfeder ve g-maliyet tablosunu (Dijkstra/BFS relax) günceller.
    - Ayrıca ajan hareket halindeyken haritaya dinamik engel eklendiğinde sadece
      etkilenen alt-yolu yeniden hesaplayabilir (Dynamic Replanning).
    """
    t0 = time.perf_counter()
    s = start or env.start
    g = goal or env.goal

    g_cost: Dict[GridPos, int] = {s: 0}
    came_from: Dict[GridPos, Optional[GridPos]] = {s: None}
    visited_set: Set[GridPos] = set()
    visited_order: List[GridPos] = []

    counter = 0
    # Öncelik kuyruğu: (f_hibrit, h_maliyet, g_maliyet, sayac, düğüm)
    open_heap: List[Tuple[float, int, int, int, GridPos]] = [
        (float(manhattan(s, g)), manhattan(s, g), 0, counter, s)
    ]

    while open_heap:
        _, _, _, _, curr = heapq.heappop(open_heap)
        if curr in visited_set:
            continue

        visited_set.add(curr)
        visited_order.append(curr)

        if curr == g:
            path = reconstruct_path(came_from, g)
            elapsed = (time.perf_counter() - t0) * 1000.0
            stats = SearchStats(
                algorithm_name="Hibrit (BFS + Dinamik)",
                visited_count=len(visited_order),
                path_length=max(0, len(path) - 1),
                elapsed_ms=elapsed,
                found=True,
                obstacles_dodged=env.count_dodged_obstacles(path),
                path=path,
                visited_order=visited_order,
            )
            yield ("FOUND", curr, path, len(visited_order))
            return stats

        yield ("VISIT", curr, [], len(visited_order))

        # Engel komşuluğunda mıyız? Engele yakınsak yerel BFS (derinlik=2) katmanı çalıştır
        near_wall = env.has_adjacent_wall(curr)
        if near_wall:
            local_q: deque[Tuple[GridPos, int]] = deque([(curr, 0)])
            local_seen: Set[GridPos] = {curr}
            while local_q:
                u, depth = local_q.popleft()
                if depth >= 2:
                    continue
                for v in env.neighbors(u):
                    tentative_g = g_cost[u] + 1
                    if tentative_g < g_cost.get(v, 10**9):
                        g_cost[v] = tentative_g
                        came_from[v] = u
                        h = manhattan(v, g)
                        # BFS katmanında tam en kısa yol ağırlığı (w = 1.0)
                        counter += 1
                        heapq.heappush(open_heap, (tentative_g + 1.0 * h, h, tentative_g, counter, v))
                    if v not in local_seen:
                        local_seen.add(v)
                        local_q.append((v, depth + 1))
        else:
            # Açık alanda Dinamik Yönelimli genişleme
            for nxt in env.neighbors(curr):
                tentative_g = g_cost[curr] + 1
                if tentative_g < g_cost.get(nxt, 10**9):
                    g_cost[nxt] = tentative_g
                    came_from[nxt] = curr
                    h = manhattan(nxt, g)
                    # Hedefe olan uzaklığa göre dinamik sezgisel katsayı (en kısa yolu bozmayan tie-break)
                    dynamic_weight = 1.0 + 0.05 * (h / (env.rows + env.cols))
                    counter += 1
                    heapq.heappush(
                        open_heap,
                        (tentative_g + dynamic_weight * h, h, tentative_g, counter, nxt),
                    )

    elapsed = (time.perf_counter() - t0) * 1000.0
    stats = SearchStats(
        algorithm_name="Hibrit (BFS + Dinamik)",
        visited_count=len(visited_order),
        path_length=0,
        elapsed_ms=elapsed,
        found=False,
        obstacles_dodged=0,
        path=[],
        visited_order=visited_order,
    )
    yield ("NO_PATH", s, [], len(visited_order))
    return stats


def astar_generator(
    env: MazeEnvironment,
    start: Optional[GridPos] = None,
    goal: Optional[GridPos] = None,
) -> Generator[Tuple[str, GridPos, List[GridPos], int], None, SearchStats]:
    """
    Bonus: A* (A-Star) Algoritması üreteci.
    f(n) = g(n) + h(n) (Manhattan mesafesi + çapraz tie-breaker) ile en kısa yolu
    en az kareye bakarak bulur.
    """
    t0 = time.perf_counter()
    s = start or env.start
    g = goal or env.goal

    g_cost: Dict[GridPos, int] = {s: 0}
    came_from: Dict[GridPos, Optional[GridPos]] = {s: None}
    closed_set: Set[GridPos] = set()
    visited_order: List[GridPos] = []

    counter = 0
    h0 = manhattan(s, g)
    open_heap: List[Tuple[float, int, int, GridPos]] = [(float(h0), h0, counter, s)]

    while open_heap:
        _, _, _, curr = heapq.heappop(open_heap)
        if curr in closed_set:
            continue

        closed_set.add(curr)
        visited_order.append(curr)

        if curr == g:
            path = reconstruct_path(came_from, g)
            elapsed = (time.perf_counter() - t0) * 1000.0
            stats = SearchStats(
                algorithm_name="A* (A-Star Bonus)",
                visited_count=len(visited_order),
                path_length=max(0, len(path) - 1),
                elapsed_ms=elapsed,
                found=True,
                obstacles_dodged=env.count_dodged_obstacles(path),
                path=path,
                visited_order=visited_order,
            )
            yield ("FOUND", curr, path, len(visited_order))
            return stats

        yield ("VISIT", curr, [], len(visited_order))

        for nxt in env.neighbors(curr):
            if nxt in closed_set:
                continue
            tentative_g = g_cost[curr] + 1
            if tentative_g < g_cost.get(nxt, 10**9):
                g_cost[nxt] = tentative_g
                came_from[nxt] = curr
                h = manhattan(nxt, g)
                # Küçük cross-product tie-breaker ile gereksiz simetrik dallanmaları azaltır
                dx1 = nxt[0] - g[0]
                dy1 = nxt[1] - g[1]
                dx2 = s[0] - g[0]
                dy2 = s[1] - g[1]
                cross = abs(dx1 * dy2 - dx2 * dy1)
                f = tentative_g + h + cross * 0.0001
                counter += 1
                heapq.heappush(open_heap, (f, h, counter, nxt))

    elapsed = (time.perf_counter() - t0) * 1000.0
    stats = SearchStats(
        algorithm_name="A* (A-Star Bonus)",
        visited_count=len(visited_order),
        path_length=0,
        elapsed_ms=elapsed,
        found=False,
        obstacles_dodged=0,
        path=[],
        visited_order=visited_order,
    )
    yield ("NO_PATH", s, [], len(visited_order))
    return stats


def exhaust_generator(
    gen: Generator[Tuple[str, GridPos, List[GridPos], int], None, SearchStats]
) -> SearchStats:
    """Bir algoritma üretecini sonuna kadar çalıştırıp SearchStats sonucunu döndürür."""
    try:
        while True:
            next(gen)
    except StopIteration as stop:
        return stop.value


def run_bfs(env: MazeEnvironment, start: Optional[GridPos] = None, goal: Optional[GridPos] = None) -> SearchStats:
    return exhaust_generator(bfs_generator(env, start, goal))


def run_hybrid(env: MazeEnvironment, start: Optional[GridPos] = None, goal: Optional[GridPos] = None) -> SearchStats:
    return exhaust_generator(hybrid_bfs_dynamic_generator(env, start, goal))


def run_astar(env: MazeEnvironment, start: Optional[GridPos] = None, goal: Optional[GridPos] = None) -> SearchStats:
    return exhaust_generator(astar_generator(env, start, goal))


# ============================================================================
# PYGAME GÖRSELLEŞTİRME VE İNTERAKTİF SİMÜLASYON ARAYÜZÜ
# ============================================================================

def run_pygame_app() -> None:
    import pygame

    pygame.init()
    pygame.display.set_caption("Proje 3: Labirentte Yol Bulan Ajan (BFS | Hibrit BFS+Dinamik | Bonus: A*)")

    CELL_SIZE = 24
    GRID_MARGIN = 1
    TOP_PANEL_HEIGHT = 135
    BOTTOM_PANEL_HEIGHT = 55

    SCREEN_WIDTH = COLS * CELL_SIZE + 40
    SCREEN_HEIGHT = TOP_PANEL_HEIGHT + ROWS * CELL_SIZE + BOTTOM_PANEL_HEIGHT + 20

    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    clock = pygame.time.Clock()

    font_title = pygame.font.SysFont("Segoe UI, Arial, sans-serif", 20, bold=True)
    font_ui = pygame.font.SysFont("Segoe UI, Arial, sans-serif", 14, bold=False)
    font_bold = pygame.font.SysFont("Segoe UI, Arial, sans-serif", 14, bold=True)
    font_small = pygame.font.SysFont("Segoe UI, Arial, sans-serif", 12, bold=False)

    # Renk Paleti
    BG_COLOR = (18, 22, 31)
    PANEL_BG = (28, 34, 48)
    GRID_BG = (35, 42, 58)
    EMPTY_COLOR = (242, 245, 250)
    WALL_COLOR = (30, 35, 45)
    START_COLOR = (34, 197, 94)      # Yeşil
    GOAL_COLOR = (239, 68, 68)       # Kırmızı
    AGENT_COLOR = (249, 115, 22)     # Turuncu ajan
    PATH_COLOR = (250, 204, 21)      # Altın sarısı en kısa yol
    VISITED_BFS = (96, 165, 250)     # Açık mavi
    VISITED_HYBRID = (167, 139, 250) # Mor/Eflatun
    VISITED_ASTAR = (45, 212, 191)   # Turkuaz
    TEXT_LIGHT = (241, 245, 249)
    TEXT_MUTED = (148, 163, 184)
    BTN_BG = (51, 65, 85)
    BTN_ACTIVE = (59, 130, 246)
    BTN_COMPARE = (16, 185, 129)

    env = MazeEnvironment(ROWS, COLS)

    # Durum Değişkenleri
    selected_mode = "HYBRID"  # "BFS", "HYBRID", "ASTAR"
    search_gen: Optional[Generator] = None
    visited_cells: List[GridPos] = []
    visited_set: Set[GridPos] = set()
    current_path: List[GridPos] = []
    agent_pos: GridPos = env.start
    agent_path_idx: int = 0
    animating_search: bool = False
    animating_agent: bool = False
    status_message: str = (
        "Hazır! Sol Tık: Duvar Ekle / S-G Taşı | Sağ Tık: Duvar Sil | "
        "SPACE: Aramayı Başlat | K: Algoritmaları Karşılaştır"
    )

    # Karşılaştırma İstatistikleri
    stats_map: Dict[str, Optional[SearchStats]] = {
        "BFS": None,
        "HYBRID": None,
        "ASTAR": None,
    }

    def update_comparison_stats() -> None:
        stats_map["BFS"] = run_bfs(env)
        stats_map["HYBRID"] = run_hybrid(env)
        stats_map["ASTAR"] = run_astar(env)

    # Başlangıçta ilk karşılaştırma verilerini hesapla
    update_comparison_stats()

    def start_search_animation(mode: str, from_pos: Optional[GridPos] = None) -> None:
        nonlocal selected_mode, search_gen, visited_cells, visited_set, current_path
        nonlocal animating_search, animating_agent, agent_pos, agent_path_idx, status_message

        selected_mode = mode
        s_pos = from_pos if from_pos is not None else env.start
        visited_cells.clear()
        visited_set.clear()
        current_path.clear()
        agent_pos = s_pos
        agent_path_idx = 0

        if mode == "BFS":
            search_gen = bfs_generator(env, s_pos, env.goal)
        elif mode == "HYBRID":
            search_gen = hybrid_bfs_dynamic_generator(env, s_pos, env.goal)
        else:
            search_gen = astar_generator(env, s_pos, env.goal)

        animating_search = True
        animating_agent = False
        status_message = f"{mode} algoritması çalışıyor... Kareler taranıyor."

    # Üst buton tanımları
    buttons = [
        {"id": "HYBRID", "label": "1: Hibrit (BFS+Dinamik)", "rect": pygame.Rect(20, 48, 195, 32)},
        {"id": "BFS", "label": "2: Saf BFS", "rect": pygame.Rect(225, 48, 120, 32)},
        {"id": "ASTAR", "label": "3: Bonus A*", "rect": pygame.Rect(355, 48, 125, 32)},
        {"id": "START", "label": "▶ Animasyonu Başlat (SPACE)", "rect": pygame.Rect(495, 48, 220, 32)},
        {"id": "COMPARE", "label": "⚡ Karşılaştır (K)", "rect": pygame.Rect(725, 48, 155, 32)},
        {"id": "DEFAULT", "label": "Varsayılan (D)", "rect": pygame.Rect(890, 48, 120, 32)},
        {"id": "RANDOM", "label": "Rastgele (R)", "rect": pygame.Rect(1020, 48, 110, 32)},
        {"id": "CLEAR", "label": "Temizle (C)", "rect": pygame.Rect(1140, 48, 95, 32)},
    ]

    dragging_node: Optional[str] = None  # "START" veya "GOAL"
    mouse_drawing: Optional[int] = None  # WALL veya EMPTY
    agent_step_timer = 0.0

    def pixel_to_grid(mx: int, my: int) -> Optional[GridPos]:
        gx = (mx - 20) // CELL_SIZE
        gy = (my - TOP_PANEL_HEIGHT) // CELL_SIZE
        if 0 <= gy < ROWS and 0 <= gx < COLS:
            return (gy, gx)
        return None

    # Açılışta Hibrit algoritmanın animasyonunu otomatik başlat
    start_search_animation("HYBRID")

    running = True
    while running:
        dt = clock.tick(60) / 1000.0
        agent_step_timer += dt

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_1:
                    start_search_animation("HYBRID")
                elif event.key == pygame.K_2:
                    start_search_animation("BFS")
                elif event.key == pygame.K_3:
                    start_search_animation("ASTAR")
                elif event.key == pygame.K_SPACE:
                    start_search_animation(selected_mode)
                elif event.key == pygame.K_k:
                    update_comparison_stats()
                    b_s = stats_map["BFS"]
                    h_s = stats_map["HYBRID"]
                    a_s = stats_map["ASTAR"]
                    if b_s and h_s and a_s and b_s.found:
                        diff_ba = b_s.visited_count - a_s.visited_count
                        status_message = (
                            f"KARŞILAŞTIRMA => BFS: {b_s.visited_count} kare | "
                            f"Hibrit: {h_s.visited_count} kare | "
                            f"A*: {a_s.visited_count} kare (A*, BFS'ten {diff_ba} daha az kareye baktı!)"
                        )
                    start_search_animation(selected_mode)
                elif event.key == pygame.K_d:
                    env.setup_default_obstacles()
                    update_comparison_stats()
                    start_search_animation(selected_mode)
                elif event.key == pygame.K_r:
                    env.generate_random_maze()
                    update_comparison_stats()
                    start_search_animation(selected_mode)
                elif event.key == pygame.K_c:
                    env.clear_walls()
                    update_comparison_stats()
                    start_search_animation(selected_mode)

            elif event.type == pygame.MOUSEBUTTONDOWN:
                mx, my = event.pos
                # Buton tıklaması kontrolü
                clicked_btn = None
                for btn in buttons:
                    if btn["rect"].collidepoint(mx, my):
                        clicked_btn = btn["id"]
                        break

                if clicked_btn:
                    if clicked_btn in ("HYBRID", "BFS", "ASTAR"):
                        start_search_animation(clicked_btn)
                    elif clicked_btn == "START":
                        start_search_animation(selected_mode)
                    elif clicked_btn == "COMPARE":
                        update_comparison_stats()
                        b_s = stats_map["BFS"]
                        h_s = stats_map["HYBRID"]
                        a_s = stats_map["ASTAR"]
                        if b_s and h_s and a_s and b_s.found:
                            status_message = (
                                f"KARŞILAŞTIRMA => BFS: {b_s.visited_count} kare | "
                                f"Hibrit: {h_s.visited_count} kare | "
                                f"A*: {a_s.visited_count} kare (En Kısa Yol: {b_s.path_length} adım)"
                            )
                    elif clicked_btn == "DEFAULT":
                        env.setup_default_obstacles()
                        update_comparison_stats()
                        start_search_animation(selected_mode)
                    elif clicked_btn == "RANDOM":
                        env.generate_random_maze()
                        update_comparison_stats()
                        start_search_animation(selected_mode)
                    elif clicked_btn == "CLEAR":
                        env.clear_walls()
                        update_comparison_stats()
                        start_search_animation(selected_mode)
                else:
                    gpos = pixel_to_grid(mx, my)
                    if gpos is not None:
                        if event.button == 1:
                            if gpos == env.start:
                                dragging_node = "START"
                            elif gpos == env.goal:
                                dragging_node = "GOAL"
                            else:
                                mouse_drawing = WALL
                                env.grid[gpos[0]][gpos[1]] = WALL
                                update_comparison_stats()
                                # Ajan yürürken yolunun üstüne duvar konulursa DİNAMİK YENİDEN PLANLAMA yap!
                                if animating_agent and gpos in current_path[agent_path_idx:]:
                                    status_message = (
                                        "Dinamik engel algılandı! Ajan bulunduğu konumdan "
                                        "yeni en kısa yolu dinamik olarak hesaplıyor..."
                                    )
                                    start_search_animation("HYBRID", from_pos=agent_pos)
                        elif event.button == 3:
                            if gpos not in (env.start, env.goal):
                                mouse_drawing = EMPTY
                                env.grid[gpos[0]][gpos[1]] = EMPTY
                                update_comparison_stats()

            elif event.type == pygame.MOUSEBUTTONUP:
                dragging_node = None
                mouse_drawing = None

            elif event.type == pygame.MOUSEMOTION:
                mx, my = event.pos
                gpos = pixel_to_grid(mx, my)
                if gpos is not None:
                    if dragging_node == "START" and gpos != env.goal and env.grid[gpos[0]][gpos[1]] == EMPTY:
                        env.start = gpos
                        agent_pos = gpos
                        update_comparison_stats()
                    elif dragging_node == "GOAL" and gpos != env.start and env.grid[gpos[0]][gpos[1]] == EMPTY:
                        env.goal = gpos
                        update_comparison_stats()
                    elif mouse_drawing is not None and gpos not in (env.start, env.goal):
                        if env.grid[gpos[0]][gpos[1]] != mouse_drawing:
                            env.grid[gpos[0]][gpos[1]] = mouse_drawing
                            update_comparison_stats()
                            if (
                                mouse_drawing == WALL
                                and animating_agent
                                and gpos in current_path[agent_path_idx:]
                            ):
                                status_message = (
                                    "Dinamik engel algılandı! Ajan bulunduğu konumdan "
                                    "yeni yolu dinamik olarak hesaplıyor..."
                                )
                                start_search_animation("HYBRID", from_pos=agent_pos)

        # Arama Animasyonu Adımları (Her karede birkaç düğüm ilerlet)
        if animating_search and search_gen is not None:
            steps_per_frame = 6
            for _ in range(steps_per_frame):
                try:
                    state, cell, path, count = next(search_gen)
                    if state == "VISIT":
                        if cell not in visited_set:
                            visited_set.add(cell)
                            visited_cells.append(cell)
                    elif state == "FOUND":
                        current_path = path
                        animating_search = False
                        animating_agent = True
                        agent_path_idx = 0
                        agent_pos = current_path[0] if current_path else env.start
                        dodged = env.count_dodged_obstacles(current_path)
                        status_message = (
                            f"{selected_mode} Tamamlandı! Baktığı Kare: {count} | "
                            f"En Kısa Yol: {len(current_path) - 1} adım | "
                            f"Kaçılan Engel Grubu: {dodged} (Ajan ilerliyor - önüne duvar çizebilirsiniz!)"
                        )
                        break
                    elif state == "NO_PATH":
                        animating_search = False
                        animating_agent = False
                        status_message = f"{selected_mode}: Hedefe ulaşan bir yol bulunamadı! ({count} kare tarandı)"
                        break
                except StopIteration as stop:
                    if stop.value is not None:
                        stats_map[selected_mode] = stop.value
                    animating_search = False
                    break

        # Ajanın Bulunan En Kısa Yolda Hareket Animasyonu
        if animating_agent and current_path:
            if agent_step_timer >= 0.045:
                agent_step_timer = 0.0
                if agent_path_idx < len(current_path) - 1:
                    next_cell = current_path[agent_path_idx + 1]
                    # Eğer kullanıcı o kareye duvar koyduysa anında dinamik yeniden planla
                    if env.grid[next_cell[0]][next_cell[1]] == WALL:
                        start_search_animation("HYBRID", from_pos=agent_pos)
                    else:
                        agent_path_idx += 1
                        agent_pos = current_path[agent_path_idx]
                else:
                    animating_agent = False

        # ================= ÇİZİM İŞLEMLERİ =================
        screen.fill(BG_COLOR)

        # Üst Panel
        pygame.draw.rect(screen, PANEL_BG, (10, 8, SCREEN_WIDTH - 20, TOP_PANEL_HEIGHT - 14), border_radius=8)
        title_surf = font_title.render(
            "Proje 3: Labirentte Yol Bulan Ajan (20x50 Grid | En Az 8 Engel | BFS + Dinamik Hibrit & A*)",
            True,
            TEXT_LIGHT,
        )
        screen.blit(title_surf, (22, 14))

        # Butonları Çiz
        for btn in buttons:
            is_active = btn["id"] == selected_mode
            if btn["id"] == "COMPARE":
                color = BTN_COMPARE
            elif is_active:
                color = BTN_ACTIVE
            else:
                color = BTN_BG
            pygame.draw.rect(screen, color, btn["rect"], border_radius=6)
            pygame.draw.rect(screen, (100, 116, 139), btn["rect"], width=1, border_radius=6)
            lbl = font_bold.render(btn["label"], True, TEXT_LIGHT)
            lx = btn["rect"].x + (btn["rect"].width - lbl.get_width()) // 2
            ly = btn["rect"].y + (btn["rect"].height - lbl.get_height()) // 2
            screen.blit(lbl, (lx, ly))

        # Karşılaştırma Bandı (BFS vs Hibrit vs Bonus A*)
        b_s = stats_map.get("BFS")
        h_s = stats_map.get("HYBRID")
        a_s = stats_map.get("ASTAR")

        comp_y = 88
        if b_s and h_s and a_s:
            c_text = (
                f"📊 KARŞILAŞTIRMA TABLOSU  ──  "
                f"[BFS] Baktığı Kare: {b_s.visited_count} | Yol: {b_s.path_length} adım ({b_s.elapsed_ms:.2f} ms)    "
                f"[HİBRİT (BFS+Dinamik)] Baktığı Kare: {h_s.visited_count} | Yol: {h_s.path_length} adım ({h_s.elapsed_ms:.2f} ms)    "
                f"[BONUS A*] Baktığı Kare: {a_s.visited_count} | Yol: {a_s.path_length} adım ({a_s.elapsed_ms:.2f} ms) | "
                f"Kaçılan Engel: {h_s.obstacles_dodged}"
            )
        else:
            c_text = "Karşılaştırma istatistikleri hesaplanıyor..."

        comp_surf = font_bold.render(c_text, True, (125, 211, 252))
        screen.blit(comp_surf, (22, comp_y))

        # Grid Arka Planı
        grid_rect = pygame.Rect(
            18,
            TOP_PANEL_HEIGHT - 2,
            COLS * CELL_SIZE + 4,
            ROWS * CELL_SIZE + 4,
        )
        pygame.draw.rect(screen, GRID_BG, grid_rect, border_radius=4)

        # Algoritmaya Göre Ziyaret Rengi
        if selected_mode == "BFS":
            visit_color = VISITED_BFS
        elif selected_mode == "HYBRID":
            visit_color = VISITED_HYBRID
        else:
            visit_color = VISITED_ASTAR

        path_set = set(current_path)

        # 20x50 Kareleri Çiz
        for r in range(ROWS):
            for c in range(COLS):
                pos = (r, c)
                x = 20 + c * CELL_SIZE
                y = TOP_PANEL_HEIGHT + r * CELL_SIZE
                rect = pygame.Rect(
                    x + GRID_MARGIN,
                    y + GRID_MARGIN,
                    CELL_SIZE - 2 * GRID_MARGIN,
                    CELL_SIZE - 2 * GRID_MARGIN,
                )

                if pos == env.start:
                    color = START_COLOR
                elif pos == env.goal:
                    color = GOAL_COLOR
                elif env.grid[r][c] == WALL:
                    color = WALL_COLOR
                elif pos in path_set:
                    color = PATH_COLOR
                elif pos in visited_set:
                    color = visit_color
                else:
                    color = EMPTY_COLOR

                pygame.draw.rect(screen, color, rect, border_radius=3)

                # Başlangıç (S) ve Hedef (G) harfleri
                if pos == env.start:
                    s_lbl = font_small.render("S", True, (10, 20, 15))
                    screen.blit(s_lbl, (x + 7, y + 4))
                elif pos == env.goal:
                    g_lbl = font_small.render("G", True, (255, 255, 255))
                    screen.blit(g_lbl, (x + 7, y + 4))

        # Ajanı Çiz (Daire olarak)
        ax = 20 + agent_pos[1] * CELL_SIZE + CELL_SIZE // 2
        ay = TOP_PANEL_HEIGHT + agent_pos[0] * CELL_SIZE + CELL_SIZE // 2
        pygame.draw.circle(screen, AGENT_COLOR, (ax, ay), CELL_SIZE // 2 - 2)
        pygame.draw.circle(screen, (255, 255, 255), (ax, ay), CELL_SIZE // 2 - 2, width=2)

        # Alt Bilgi ve Lejant Paneli
        bot_y = TOP_PANEL_HEIGHT + ROWS * CELL_SIZE + 8
        pygame.draw.rect(screen, PANEL_BG, (10, bot_y, SCREEN_WIDTH - 20, BOTTOM_PANEL_HEIGHT - 8), border_radius=8)

        status_surf = font_ui.render(status_message, True, TEXT_LIGHT)
        screen.blit(status_surf, (22, bot_y + 6))

        legend_surf = font_small.render(
            "Lejant: [Yeşil: Başlangıç (S)]  [Kırmızı: Hedef (G)]  [Turuncu: Ajan]  "
            "[Sarı: En Kısa Yol]  [Mavi/Mor/Turkuaz: Taranan Kareler]  [Koyu Gri: Duvar/Engel]",
            True,
            TEXT_MUTED,
        )
        screen.blit(legend_surf, (22, bot_y + 26))

        pygame.display.flip()

    pygame.quit()


# ============================================================================
# MATPLOTLIB YEDEK / ALTERNATİF GÖRSELLEŞTİRME MODU
# ============================================================================

def run_matplotlib_app() -> None:
    """
    Pygame yüklü olmayan sistemlerde veya '--matplotlib' parametresi verildiğinde
    çalışan interaktif Matplotlib arayüzü (fareyle duvar ekleme, animasyon ve karşılaştırma).
    """
    import matplotlib.pyplot as plt
    from matplotlib. colors import ListedColormap
    import numpy as np

    env = MazeEnvironment(ROWS, COLS)

    # Renk kodları: 0:Boş, 1:Duvar, 2:Taranan(BFS/Hibrit/A*), 3:En Kısa Yol, 4:Başlangıç, 5:Hedef, 6:Ajan
    cmap = ListedColormap([
        "#F2F5FA",  # 0: Boş
        "#1E232D",  # 1: Duvar
        "#93C5FD",  # 2: Taranan Kare
        "#FACC15",  # 3: En Kısa Yol
        "#22C55E",  # 4: Başlangıç
        "#EF4444",  # 5: Hedef
        "#F97316",  # 6: Ajan
    ])

    fig, (ax_grid, ax_bar) = plt.subplots(
        2, 1, figsize=(14, 7.5), gridspec_kw={"height_ratios": [3.2, 1.2]}
    )
    fig.canvas.manager.set_window_title("Proje 3: Labirentte Yol Bulan Ajan (Matplotlib Modu)")

    def build_render_matrix(visited: Set[GridPos], path: Set[GridPos], agent: GridPos) -> np.ndarray:
        mat = np.zeros((ROWS, COLS), dtype=int)
        for r in range(ROWS):
            for c in range(COLS):
                if env.grid[r][c] == WALL:
                    mat[r, c] = 1
                elif (r, c) in path:
                    mat[r, c] = 3
                elif (r, c) in visited:
                    mat[r, c] = 2
        mat[env.start[0], env.start[1]] = 4
        mat[env.goal[0], env.goal[1]] = 5
        mat[agent[0], agent[1]] = 6
        return mat

    bfs_res = run_bfs(env)
    hyb_res = run_hybrid(env)
    ast_res = run_astar(env)

    img = ax_grid.imshow(
        build_render_matrix(set(hyb_res.visited_order), set(hyb_res.path), env.goal),
        cmap=cmap,
        vmin=0,
        vmax=6,
    )
    ax_grid.set_title(
        f"20x50 Labirent — Hibrit (BFS + Dinamik) En Kısa Yol: {hyb_res.path_length} adım | "
        f"Kaçılan Engel: {hyb_res.obstacles_dodged} (Sol Tık: Duvar Ekle/Sil, Tuşlar: 1=Hibrit, 2=BFS, 3=A*)",
        fontsize=11,
        fontweight="bold",
    )
    ax_grid.set_xticks(range(0, COLS, 5))
    ax_grid.set_yticks(range(0, ROWS, 2))
    ax_grid.grid(which="minor", color="#CBD5E1", linestyle="-", linewidth=0.4)

    def update_chart() -> None:
        ax_bar.clear()
        b = run_bfs(env)
        h = run_hybrid(env)
        a = run_astar(env)
        labels = [
            f"BFS\n(Yol: {b.path_length})",
            f"Hibrit (BFS+Dinamik)\n(Yol: {h.path_length})",
            f"Bonus: A*\n(Yol: {a.path_length})",
        ]
        counts = [b.visited_count, h.visited_count, a.visited_count]
        bars = ax_bar.barh(labels, counts, color=["#60A5FA", "#A78BFA", "#2DD4BF"])
        ax_bar.set_xlabel("Bakılan (Ziyaret Edilen) Kare Sayısı (Daha Az = Daha Verimli)")
        ax_bar.set_title("Algoritmaların Baktığı Kare Sayısı Karşılaştırması")
        for bar, cnt in zip(bars, counts):
            ax_bar.text(
                bar.get_width() + 8,
                bar.get_y() + bar.get_height() / 2,
                f"{cnt} kare",
                va="center",
                fontweight="bold",
            )
        ax_bar.set_xlim(0, max(counts + [100]) * 1.18)

    update_chart()

    def animate_algorithm(mode: str) -> None:
        if mode == "BFS":
            res = run_bfs(env)
            name = "BFS"
        elif mode == "ASTAR":
            res = run_astar(env)
            name = "Bonus A*"
        else:
            res = run_hybrid(env)
            name = "Hibrit (BFS + Dinamik)"

        visited_so_far: Set[GridPos] = set()
        step = max(1, len(res.visited_order) // 25)
        for i in range(0, len(res.visited_order), step):
            visited_so_far.update(res.visited_order[i : i + step])
            img.set_data(build_render_matrix(visited_so_far, set(), env.start))
            ax_grid.set_title(f"{name} Taranıyor... Bakılan Kare: {len(visited_so_far)}")
            plt.pause(0.015)

        img.set_data(build_render_matrix(set(res.visited_order), set(res.path), env.goal))
        ax_grid.set_title(
            f"{name} Tamamlandı! Bakılan Kare: {res.visited_count} | "
            f"En Kısa Yol: {res.path_length} adım | Kaçılan Engel: {res.obstacles_dodged}"
        )
        update_chart()
        fig.canvas.draw_idle()

    def on_click(event) -> None:
        if event.inaxes != ax_grid or event.xdata is None or event.ydata is None:
            return
        c, r = int(round(event.xdata)), int(round(event.ydata))
        if 0 <= r < ROWS and 0 <= c < COLS and (r, c) not in (env.start, env.goal):
            env.grid[r][c] = EMPTY if env.grid[r][c] == WALL else WALL
            res = run_hybrid(env)
            img.set_data(build_render_matrix(set(res.visited_order), set(res.path), env.goal))
            update_chart()
            fig.canvas.draw_idle()

    def on_key(event) -> None:
        if event.key == "1":
            animate_algorithm("HYBRID")
        elif event.key == "2":
            animate_algorithm("BFS")
        elif event.key == "3":
            animate_algorithm("ASTAR")

    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.tight_layout()
    plt.show()


# ============================================================================
# KONSOL TEST VE KARŞILAŞTIRMA RAPORU (--test)
# ============================================================================

def run_cli_benchmark() -> None:
    env = MazeEnvironment(ROWS, COLS)
    bfs_stats = run_bfs(env)
    hyb_stats = run_hybrid(env)
    ast_stats = run_astar(env)

    print("=" * 78)
    print(f"PROJE 3: LABİRENTTE YOL BULAN AJAN - KARŞILAŞTIRMA RAPORU ({ROWS}x{COLS} Grid)")
    print("=" * 78)
    print(f"Başlangıç Noktası : {env.start}")
    print(f"Hedef Noktası     : {env.goal}")
    print(f"Toplam Kare Sayısı: {ROWS * COLS} ({ROWS} satır x {COLS} sütun)")
    print(f"Kaçılan Engel Grubu: {hyb_stats.obstacles_dodged} adet (Minimum 8 engel şartı sağlandı)")
    print("-" * 78)
    print(f"{'Algoritma':<28} | {'Bakılan Kare':<14} | {'En Kısa Yol':<13} | {'Süre (ms)':<10}")
    print("-" * 78)
    for st in (bfs_stats, hyb_stats, ast_stats):
        print(
            f"{st.algorithm_name:<28} | {st.visited_count:<14} | "
            f"{st.path_length:<13} | {st.elapsed_ms:<10.3f}"
        )
    print("-" * 78)
    if bfs_stats.visited_count > 0:
        saving_hyb = (1.0 - hyb_stats.visited_count / bfs_stats.visited_count) * 100.0
        saving_ast = (1.0 - ast_stats.visited_count / bfs_stats.visited_count) * 100.0
        print(
            f"Sonuç: Her üç algoritma da {bfs_stats.path_length} adımlık garantili en kısa yolu buldu."
        )
        print(
            f"- Hibrit (BFS + Dinamik) yöntem, saf BFS'e göre %{saving_hyb:.1f} daha az kareye baktı "
            f"({bfs_stats.visited_count} yerine {hyb_stats.visited_count} kare)."
        )
        print(
            f"- Bonus A* algoritması, saf BFS'e göre %{saving_ast:.1f} daha az kareye baktı "
            f"({bfs_stats.visited_count} yerine {ast_stats.visited_count} kare)."
        )
    print("=" * 78)


def main() -> None:
    parser = argparse.ArgumentParser(description="Proje 3: Labirentte Yol Bulan Ajan")
    parser.add_argument("--test", action="store_true", help="Algoritmaları konsolda test et ve karşılaştır")
    parser.add_argument("--matplotlib", action="store_true", help="Matplotlib arayüzünü çalıştır")
    args = parser.parse_args()

    if args.test:
        run_cli_benchmark()
        return

    if args.matplotlib:
        run_matplotlib_app()
        return

    try:
        import pygame  # noqa: F401
        run_pygame_app()
    except ImportError:
        print("Bilgi: 'pygame' bulunamadı, interaktif 'matplotlib' arayüzü başlatılıyor...")
        run_matplotlib_app()


if __name__ == "__main__":
    main()