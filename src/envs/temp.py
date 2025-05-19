# t = val  # fade from center to boundary
# alpha = int(200 * (1.0 - t))  # semi-transparent, not max
# # Blend color between light and dark green
# r = int(light[0] * (1 - t) + dark[0] * t)
# g = int(light[1] * (1 - t) + dark[1] * t)
# b = int(light[2] * (1 - t) + dark[2] * t)
# # Clamp
# r = max(0, min(255, r))
# g = max(0, min(255, g))
# b = max(0, min(255, b))
# alpha = max(0, min(255, alpha))
# surf.set_at((px, py), (r, g, b, alpha))
