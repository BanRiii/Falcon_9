import pygame
import numpy as np
import random
import sys

# ============================================================
# 1. Parámetros físicos del cohete (iguales al código del profe)
# ============================================================
m = 1500.0
g = 9.81
l = 10.0
I = 12000.0
F_nominal = m * g

MAX_GIMBAL_DEG = 35.0
max_gimbal_rad = np.radians(MAX_GIMBAL_DEG)
MAX_THRUST = 3.5 * F_nominal
MIN_THRUST = 0.2 * F_nominal

A_FRENADO = 3.5      # Aceleración de frenado del perfil de aterrizaje (m/s^2)
V_CONTACTO = 0.6     # Velocidad mínima de descenso al acercarse al suelo (m/s)


# ============================================================
# 2. CONTROLADOR DIFUSO TIPO MAMDANI (implementado con numpy)
#    Entradas : error (e) y razón de cambio de la variable (d)
#    Salida   : acción de control
#    Conjuntos: NG, NP, Z, PP, PG  (triangulares, con hombros en los extremos)
#    Reglas   : tabla 5x5, AND = mínimo, agregación = máximo,
#               desdifusificación = centroide
# ============================================================
ETIQUETAS = ["NG", "NP", "Z", "PP", "PG"]


class ControladorDifuso:
    def __init__(self, rango_e, rango_d, centros_salida):
        self.ce = np.linspace(-rango_e, rango_e, 5)   # centros de los conjuntos de e
        self.cd = np.linspace(-rango_d, rango_d, 5)   # centros de los conjuntos de d
        self.cs = np.array(centros_salida, dtype=float)
        base = np.eye(5)

        # Universo de salida y funciones de pertenencia de la salida
        margen = 0.15 * (self.cs[-1] - self.cs[0])
        self.u = np.linspace(self.cs[0] - margen, self.cs[-1] + margen, 401)
        self.mu_salida = [np.interp(self.u, self.cs, base[k]) for k in range(5)]
        self.base = base

        # Base de reglas: salida = i - (j - 2)   (tipo PD difuso)
        #   i = conjunto del error, j = conjunto de la derivada
        self.reglas = np.zeros((5, 5), dtype=int)
        for i in range(5):
            for j in range(5):
                self.reglas[i, j] = int(np.clip(i - (j - 2), 0, 4))

    def fuzzificar(self, valor, centros):
        return np.array([np.interp(valor, centros, self.base[k]) for k in range(5)])

    def calcular(self, e, d):
        mu_e = self.fuzzificar(e, self.ce)
        mu_d = self.fuzzificar(d, self.cd)

        # Inferencia: fuerza de disparo por conjunto de salida
        fuerza = np.zeros(5)
        for i in range(5):
            for j in range(5):
                w = min(mu_e[i], mu_d[j])                 # AND = mínimo
                k = self.reglas[i, j]
                fuerza[k] = max(fuerza[k], w)             # OR  = máximo

        # Agregación (Mamdani: recorte con mínimo, unión con máximo)
        agregada = np.zeros_like(self.u)
        for k in range(5):
            agregada = np.maximum(agregada, np.minimum(fuerza[k], self.mu_salida[k]))

        # Desdifusificación por centroide
        area = agregada.sum()
        if area < 1e-9:
            return float(self.cs[2])
        return float((self.u * agregada).sum() / area)


# --- Tres controladores difusos en cascada ---
# (a) Empuje: e = vy_objetivo - vy (m/s),  d = aceleración vertical (m/s^2)
#     salida = variación del empuje respecto al de hover (múltiplo de F_nominal)
fuzzy_empuje = ControladorDifuso(rango_e=1.5, rango_d=6.0,
                                 centros_salida=[-0.8, -0.35, 0.0, 0.8, 2.0])

# (b) Posición lateral: e = x_objetivo - x (m),  d = vx (m/s)
#     salida = inclinación deseada del cohete (grados)
fuzzy_posicion = ControladorDifuso(rango_e=40.0, rango_d=12.0,
                                   centros_salida=[-20.0, -8.0, 0.0, 8.0, 20.0])

# (c) Actitud: e = theta_deseado - theta (grados),  d = omega (grados/s)
#     salida = ángulo de la tobera (grados)
fuzzy_actitud = ControladorDifuso(rango_e=30.0, rango_d=40.0,
                                  centros_salida=[-35.0, -12.0, 0.0, 12.0, 35.0])


_filtro = {"a": 0.0}   # aceleración vertical filtrada (evita oscilaciones del empuje)


def controlar(estado, objetivo_x, vy_objetivo, y_objetivo, aceleracion_y, altura_base, aterrizando):
    """Devuelve (gimbal_rad, empuje_N) usando únicamente lógica difusa."""
    x, vx, y, vy, theta, omega = estado
    _filtro["a"] += 0.08 * (aceleracion_y - _filtro["a"])   # filtro pasa-bajas
    aceleracion_y = _filtro["a"]

    # (b) Posición lateral -> inclinación deseada
    theta_des = fuzzy_posicion.calcular(objetivo_x - x, vx)
    if aterrizando:
        # Cerca del suelo el cohete debe quedar vertical
        theta_des *= float(np.clip(altura_base / 15.0, 0.0, 1.0))

    # (c) Actitud -> ángulo de la tobera
    err_theta = theta_des - np.degrees(theta)
    gimbal_deg = fuzzy_actitud.calcular(err_theta, np.degrees(omega))
    gimbal = float(np.clip(np.radians(gimbal_deg), -max_gimbal_rad, max_gimbal_rad))

    # (a) Empuje vertical
    delta = fuzzy_empuje.calcular(vy_objetivo - vy, aceleracion_y)
    comp_incl = max(np.cos(theta + gimbal), 0.5)       # compensa la inclinación
    empuje = F_nominal * (1.0 + delta) / comp_incl
    empuje = float(np.clip(empuje, MIN_THRUST, MAX_THRUST))
    return gimbal, empuje


# ============================================================
# 3. Partículas del motor
# ============================================================
class Particula:
    def __init__(self, x, y, vx, vy):
        self.x = x
        self.y = y
        self.vx = vx + random.uniform(-15, 15)
        self.vy = vy + random.uniform(-15, 15)
        self.vida = 255
        self.color = [255, random.randint(50, 150), 0]

    def actualizar(self):
        self.x += self.vx * 0.016
        self.y += self.vy * 0.016
        self.vida -= 8

    def dibujar(self, superficie):
        if self.vida > 0:
            surf_part = pygame.Surface((6, 6), pygame.SRCALPHA)
            color_con_alfa = self.color + [self.vida]
            pygame.draw.circle(surf_part, color_con_alfa, (3, 3), 3)
            superficie.blit(surf_part, (int(self.x) - 3, int(self.y) - 3))


# ============================================================
# 4. Física (mismo modelo no lineal del código original)
# ============================================================
def integrar(x_estado, gimbal, empuje, dt):
    theta = x_estado[4]
    omega = x_estado[5]
    x_dot = np.array([
        x_estado[1],
        (empuje / m) * np.sin(theta + gimbal),
        x_estado[3],
        (empuje / m) * np.cos(theta + gimbal) - g,
        omega,
        (empuje * l / I) * np.sin(gimbal),
    ])
    nuevo = x_estado + x_dot * dt
    nuevo[4] = (nuevo[4] + np.pi) % (2 * np.pi) - np.pi
    return nuevo


def objetivo_vertical(modo, y_cm, mouse_y_fisico):
    """Velocidad vertical deseada según el modo de vuelo."""
    if modo == "LEVITACION":
        return float(np.clip(0.5 * (mouse_y_fisico - y_cm), -8.0, 8.0))
    altura_base = max(0.0, y_cm - l)
    return -max(np.sqrt(2 * A_FRENADO * altura_base), V_CONTACTO)


# ============================================================
# 5. Programa principal
# ============================================================
def main():
    pygame.init()
    ANCHO, ALTO = 1000, 800
    pantalla = pygame.display.set_mode((ANCHO, ALTO))
    pygame.display.set_caption("Control Difuso (Mamdani) - Aterrizaje Cohete Falcon 9")
    reloj = pygame.time.Clock()
    PXM = 5

    altura_cohete = 2 * l
    base_cohete = altura_cohete / 4
    fuente = pygame.font.SysFont("Consolas", 16)
    fuente_grande = pygame.font.SysFont("Consolas", 24, bold=True)

    def estado_inicial():
        return np.array([ANCHO / (2 * PXM), 0.0, ALTO / (2 * PXM), 0.0, 0.0, 0.0])

    x_estado = estado_inicial()
    particulas = []
    modo_vuelo = "LEVITACION"
    estado_impacto = ""
    vy_prev = 0.0
    gimbal_real = 0.0
    thrust_real = F_nominal
    dt = 0.016

    ejecutando = True
    while ejecutando:
        for evento in pygame.event.get():
            if evento.type == pygame.QUIT:
                ejecutando = False
            elif evento.type == pygame.KEYDOWN:
                if evento.key == pygame.K_SPACE:
                    if modo_vuelo == "LEVITACION":
                        modo_vuelo = "ATERRIZAJE"
                        estado_impacto = ""
                    else:
                        modo_vuelo = "LEVITACION"
                elif evento.key == pygame.K_r:       # reiniciar
                    x_estado = estado_inicial()
                    estado_impacto = ""
                    vy_prev = 0.0

        mouse_x, mouse_y = pygame.mouse.get_pos()
        target_x = mouse_x / PXM
        target_y = (ALTO - mouse_y) / PXM

        # ---- Control difuso ----
        y_cm = x_estado[2]
        altura_base = max(0.0, y_cm - l)
        vy_obj = objetivo_vertical(modo_vuelo, y_cm, target_y)
        acel_y = (x_estado[3] - vy_prev) / dt
        vy_prev = x_estado[3]

        gimbal_real, thrust_real = controlar(
            x_estado, target_x, vy_obj, target_y, acel_y,
            altura_base, modo_vuelo == "ATERRIZAJE")

        # ---- Física ----
        x_estado = integrar(x_estado, gimbal_real, thrust_real, dt)
        theta = x_estado[4]

        # ---- Contacto con el suelo ----
        if x_estado[2] <= l:
            if modo_vuelo == "ATERRIZAJE" and estado_impacto == "":
                v_imp = abs(x_estado[3])
                if v_imp > 4.5:
                    estado_impacto = f"¡CATASTROFE! Impacto violento a {v_imp:.2f} m/s"
                else:
                    estado_impacto = f"¡ATERRIZAJE EXITOSO! Tocado suave a {v_imp:.2f} m/s"
            if modo_vuelo == "ATERRIZAJE":
                x_estado[1] = 0.0
                x_estado[2] = l
                x_estado[3] = 0.0
                thrust_real = 0.0
                x_estado[4] = 0.0
                x_estado[5] = 0.0
                theta = 0.0
            else:
                x_estado[2] = max(x_estado[2], l)
                x_estado[3] = max(x_estado[3], 0.0)

        cx_p = x_estado[0] * PXM
        cy_p = ALTO - x_estado[2] * PXM

        cos_t, sin_t = np.cos(-theta), np.sin(-theta)

        def transformar_punto(pt):
            rx = pt[0] * cos_t - pt[1] * sin_t
            ry = pt[0] * sin_t + pt[1] * cos_t
            return [int(cx_p + rx * PXM), int(cy_p + ry * PXM)]

        punta = transformar_punto(np.array([0, -l]))
        base_izq = transformar_punto(np.array([-base_cohete / 2, l]))
        base_der = transformar_punto(np.array([base_cohete / 2, l]))
        base_centro = transformar_punto(np.array([0, l]))

        # ---- Partículas ----
        porcentaje_potencia = max(0.0, (thrust_real - MIN_THRUST) / (MAX_THRUST - MIN_THRUST))
        num_particulas = int(porcentaje_potencia * 12)
        if thrust_real > 0:
            angulo_escape = -(theta + gimbal_real) + np.pi / 2
            v_escape = 450 * (0.4 + 0.6 * porcentaje_potencia)
            for _ in range(num_particulas):
                vx_p = v_escape * np.cos(angulo_escape) + (x_estado[1] * PXM)
                vy_p = v_escape * np.sin(angulo_escape) - (x_estado[3] * PXM)
                particulas.append(Particula(base_centro[0], base_centro[1], vx_p, vy_p))

        # ---- Render ----
        pantalla.fill((15, 20, 35))
        pygame.draw.line(pantalla, (60, 70, 90), (0, ALTO - 1), (ANCHO, ALTO - 1), 2)

        if modo_vuelo == "LEVITACION":
            pygame.draw.line(pantalla, (0, 255, 0), (mouse_x - 10, mouse_y), (mouse_x + 10, mouse_y), 1)
            pygame.draw.line(pantalla, (0, 255, 0), (mouse_x, mouse_y - 10), (mouse_x, mouse_y + 10), 1)
        else:
            pygame.draw.line(pantalla, (200, 50, 50), (mouse_x, ALTO), (mouse_x, ALTO - 30), 3)

        for p in particulas[:]:
            p.actualizar()
            p.dibujar(pantalla)
            if p.vida <= 0:
                particulas.remove(p)

        pygame.draw.polygon(pantalla, (220, 225, 230), [punta, base_izq, base_der])
        pygame.draw.polygon(pantalla, (180, 190, 200), [punta, base_izq, base_der], 3)

        cos_g, sin_g = np.cos(-(theta + gimbal_real)), np.sin(-(theta + gimbal_real))
        fin_tobera = [int(base_centro[0] + sin_g * 15), int(base_centro[1] + cos_g * 15)]
        pygame.draw.line(pantalla, (255, 0, 0), base_centro, fin_tobera, 4)

        color_modo = (0, 255, 100) if modo_vuelo == "LEVITACION" else (255, 150, 0)
        pantalla.blit(fuente_grande.render(f"MODO: {modo_vuelo} (CONTROL DIFUSO)", True, color_modo), (20, 20))
        pantalla.blit(fuente.render("[ESPACIO] Cambiar modo   [R] Reiniciar", True, (150, 160, 175)), (20, 50))

        blanco = (255, 255, 255)
        pantalla.blit(fuente.render(f"Altura (Y): {x_estado[2]:.2f} m", True, blanco), (20, 90))
        pantalla.blit(fuente.render(f"Velocidad vertical: {x_estado[3]:.2f} m/s (objetivo {vy_obj:.2f})", True, blanco), (20, 110))
        pantalla.blit(fuente.render(f"Posicion lateral (X): {x_estado[0]:.2f} m", True, blanco), (20, 130))
        pantalla.blit(fuente.render(f"Inclinacion: {np.degrees(theta):.1f}°", True, blanco), (20, 150))
        pantalla.blit(fuente.render(f"Tobera (Gimbal): {np.degrees(gimbal_real):.1f}°", True, blanco), (20, 170))
        pantalla.blit(fuente.render(f"Empuje: {thrust_real/1000.0:.2f} kN ({int(porcentaje_potencia*100)}%)", True, blanco), (20, 190))

        if estado_impacto != "":
            color_res = (0, 255, 0) if "EXITOSO" in estado_impacto else (255, 0, 0)
            pantalla.blit(fuente_grande.render(estado_impacto, True, color_res), (20, 230))

        pygame.display.flip()
        reloj.tick(60)

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()