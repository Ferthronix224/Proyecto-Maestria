import cupy as cp

class Transformations:
  
  def translate(self, coor, pixel):
    '''
    Function for tracking interest points in translated images.

    Parameters
    ----------
    coor (array): Interest points array.
    pixel (int): Pixel value.
    '''
    return coor + pixel

  def rotation(self, coor, angle):
      """
      Function for tracking interest points in rotated images.

      Parameters
      ----------
      coor (array): Interest points array.
      angle (int): Rotation angle.
      """
      # Convert angle to radians.
      angle_radians = cp.radians(angle)

      # Rotation matrix.
      R = cp.array([[cp.cos(angle_radians), -cp.sin(angle_radians)],
                    [cp.sin(angle_radians), cp.cos(angle_radians)]])
      
      del angle_radians

      # Set the center.
      center = cp.array([128, 128])

      coor -= center

      # Apply rotation.
      coor @= R.T      
      del R

      coor += center
      del center

      return coor

  def scale(self, coor, scale):
      """
      Function for tracking interest points in scaled images.

      Parameters
      ----------      
      coor (array): Interest points array.
      scale (int): Scaling value.
      """
      coor = coor.astype(float)

      # Set the center.
      centro = cp.array([128, 128])
      
      coor -= centro

      # Apply scaling.
      coor *= scale / 100

      coor += centro
      del centro

      return cp.array(coor.astype(int))