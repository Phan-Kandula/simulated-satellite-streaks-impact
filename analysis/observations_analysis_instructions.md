The file data/Observations_in_field.csv contains a modified version of the observation data obtained from Slingshot Aerospace. Each row is an observation of a V2 Starlink satellite by the Slingshot Aerospace telescopes. Observations that fall outside the LSST observation range, such as those taken when the Sun's altitude is higher than -12° are removed. 


The column definitions are as follows:

| Column Name | Defintion                                                                                                                                                                                                        |
|-------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Sun Azimuth | Sun azimuth from the observer's position (deg)                                                                                                                                                                   |
| Sat Azimith | Satellite azimuth from the observer’s position (deg)                                                                                                                                                             |
| Sat Height  | Satellite Height (m)                                                                                                                                                                                             |
| Irradiance  | Satellite brightness flux (W/m^2)                                                                                                                                                                                |
| Measured AB | Satellite Tracked AB magnitude                                                                                                                                                                                   |
| AOI         | The angle past terminator. In our simulation, we model the satellite chassis parallel to the ground so that the angle past terminator will equal the angle of incidence when the sun ray hits the chassis. (deg) |
| az diff     | Azimuth difference between the sun and satellite (deg)                                                                                                                                                           |

