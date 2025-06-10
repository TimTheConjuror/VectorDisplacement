This little project is a way to generate vector displacement maps from meshes in maya.  The maps can then be applied to a vertex shader in engine to re-create whatever offsets you have sampled.

Vector displacement can be used to add life to a static mesh, replace expensive cloth sims, etc.

example:

normalize_factor = vector_displacement.create_displacement_map('my_base_geo', 'my_target_geo', 'C:/displacementPath/my_target_geo.exr', (1024, 1024), pixel_extend=2)

![example_vector_color](https://github.com/user-attachments/assets/947ada0a-68c8-4f7e-aba4-d9922f787e68)

In maya you'll see the target has been given a vertex color set, and you'll have a new image that matches, cooked out in the 'map1' uv coordinates.

![angle_up_disp](https://github.com/user-attachments/assets/399f2c7d-636a-4670-ba61-892d09a30040)

the returned value, or normalization factor, is how many times the values had to be divided by 10 to fit in the bandwidth of the color image. 1=10, 2=100, 3=1000. 
To reproduce the proper dispalcement,  subtract 0.5 form the color value, to center it, then multiply by the recorded factor of 10.

![unrealNodes](https://github.com/user-attachments/assets/7a8930fe-a63f-4b8a-85f5-ccf9b1e5286b)

Enjoy!

  Tim
