"""Synthetic research material/tool coefficients; hardness indices are NOT used."""
import math
MAT={'AL6061':(.00270,1.25,1.0),'C45':(.00785,3.2,2.1),'POM':(.00141,.72,.40),
     'WOOD':(.00075,.52,.25),'CAST_IRON':(.00720,2.8,1.8)}
TOOL={'HSS':(1.0,1.6),'CARBIDE':(.90,.70),'COATED_CARBIDE':(.86,.50),
      'CERAMIC':(.92,.65),'PCD':(.80,.35)}
def move_time(distance,velocity,accel):
    if distance <= 0:return 0.
    v=max(velocity,.01);a=max(accel,1.);critical=v*v/a
    return 2*math.sqrt(distance/a) if distance<critical else 2*v/a+(distance-critical)/v
