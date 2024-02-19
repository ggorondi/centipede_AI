import numpy as np
from scipy.signal import convolve2d

class StateExtractor:
    
    def __init__(self):

        self.past_player_pos=np.zeros(2)

        # Estos son para kernels que encuentran al player, centipede y spider
        self.kernel_list=[]
        self.kernel_vals=[]
        # Estos son para kernels que solo se usaron para entrenar (para encontrar el frame en el que muere)
        self.extra_kernel_list=[] 
        self.extra_kernel_vals=[]

        # Encuentra a Player
        kernel_player_long = np.array([ [ 1, 1, 1, 1],
                                        [ 1, 1, 1, 1],
                                        [ 1, 1, 1, 1],
                                        [ 1, 1, 1, 1],
                                        [ 1, 1, 1, 1],
                                        [ 1, 1, 1, 1],
                                        [ 1, 1, 1, 1],
                                        [ 1, 1, 1, 1],
                                        [ 1, 1, 1, 1]])
        self.kernel_list.append(kernel_player_long)
        self.kernel_vals.append(255*9*4)

        # Encuentra pedazos de centipede
        kernel_cent_piece = np.array([  [-1, 1,-1],  
                                        [ 1, 1, 1], 
                                        [ 1, 1, 1], 
                                        [ 1, 1, 1], 
                                        [ 1, 1, 1], 
                                        [-1, 1,-1]])
        self.kernel_list.append(kernel_cent_piece)
        self.kernel_vals.append(255*14)

        # Encuentra cabezas de centipede
        kernel_cent_fat = np.array([    [-1, 1, 1,-1], 
                                        [ 1, 1, 1, 1], 
                                        [ 1, 1, 1, 1], 
                                        [ 1, 1, 1, 1], 
                                        [ 1, 1, 1, 1], 
                                        [-1, 1, 1,-1]])
        self.kernel_list.append(kernel_cent_fat)
        self.kernel_vals.append(255*20)

        # Encuentra spider
        kernel_spider = np.array([      [ 1,-1,-1, 1],  #tener en cuenta que se rotan 180 grados 
                                        [ 1, 1, 1, 1], 
                                        [ 1, 1, 1, 1], 
                                        [ 1, 1, 1, 1], 
                                        [-1, 1, 1,-1]])
        self.kernel_list.append(kernel_spider)
        self.kernel_vals.append(255*16)

        # Encuentra spider
        kernel_spider_down = np.array([ [-1, 1, 1,-1],  
                                        [ 1, 1, 1, 1], 
                                        [ 1, 1, 1, 1], 
                                        [ 1, 1, 1, 1], 
                                        [ 1,-1,-1, 1]])
        self.kernel_list.append(kernel_spider_down)
        self.kernel_vals.append(255*16)

        # Encuentra spider
        kernel_spider_3 = np.array([    [ 1, 1, 1, 1],  
                                        [ 1, 1, 1, 1], 
                                        [ 1, 1, 1, 1], 
                                        [ 1, 1, 1, 1], 
                                        [-1, 1, 1,-1]])
        self.kernel_list.append(kernel_spider_3)
        self.kernel_vals.append(255*18)

        # Encuentra spider
        kernel_spider_4 = np.array([    [ 1,-1, 1, 1,-1, 1],  
                                        [ 1, 1, 1, 1, 1, 1], 
                                        [-1, 1, 1, 1, 1,-1], 
                                        [-1, 1, 1, 1, 1,-1], 
                                        [ 1, 1, 1, 1, 1, 1]])
        self.kernel_list.append(kernel_spider_4)
        self.kernel_vals.append(255*24)

        # Encuentra bloques (alfinal no los uso)
        kernel_bloque = np.array([      [-1,-1,-1,-1,-1, -1], 
                                        [-1, 1, 1, 1, 1, -1],
                                        [-1, 1, 1, 1, 1, -1],
                                        [-1, 1, 1, 1, 1, -1],
                                        [-1,-1,-1,-1,-1, -1]])
        self.kernel_list.append(kernel_bloque)
        self.kernel_vals.append(255*12)

        # Encuentra los pixeles de cuando pierde vida el player
        kernel_muerte_1 = np.array([    [ 1,-1], 
                                        [-1, 1],
                                        [ 1,-1], 
                                        [-1, 1],
                                        [ 1,-1], 
                                        [-1, 1]])
        self.extra_kernel_list.append(kernel_muerte_1)
        self.extra_kernel_vals.append(255*6)

        # Encuentra los pixeles de cuando pierde vida el player
        kernel_muerte_2 = np.array([    [-1, 1], 
                                        [-1,-1],
                                        [ 1,-1], 
                                        [-1, 1],
                                        [ 1,-1], 
                                        [-1, 1]])
        self.extra_kernel_list.append(kernel_muerte_2)
        self.extra_kernel_vals.append(255*5)

        # Encuentra los pixeles de cuando pierde vida el player
        kernel_muerte_3 = np.array([    [ 1,-1,-1,-1], 
                                        [-1, 1,-1, 1],
                                        [-1,-1, 1, 1], 
                                        [-1,-1, 1, 1],
                                        [ 1,-1, 1, 1], 
                                        [-1, 1,-1, 1]])
        self.extra_kernel_list.append(kernel_muerte_3)
        self.extra_kernel_vals.append(255*12)
    
    def extract(self,obs):
        """
        Devuelve una representación del estado del juego pequeña. 

        Esta representación es un np.array de 22 numeros. 
        Son 11 pares de coordenadas (fila,columna) de los actores importantes del juego:
        1 par del jugador, 9 pares de pedazos del centipede, y 1 par de la araña (en ese orden).
        """
        obs_gris= self.rgb_to_grayscale2(obs)
        
        # Primero convoluciono la imagen del estado del juego con kernels que encajan con los actores mas relevantes del centipede
        matrices=[]
        for kernel,value in zip(self.kernel_list,self.kernel_vals):
            convolved_image = convolve2d(obs_gris, kernel, mode='same')
            filtered_matrix = ((convolved_image == value) * 255).astype(np.uint8) #matriz va tener '255' donde esta lo que busca el kernel, y 0 else
            matrices.append(filtered_matrix)
        # Luego encuentro las posiciones activadas por la convolución para extraer coordenadas
        return self.smart_flatten_short(matrices)
    
    def extract_extra(self,obs): 
        """
        Devuelve True si perdió una vida el jugador, False si no.

        Busca los pixeles de cuando pierde una vida el jugador 
        (Lo usé para entrenar, ya que el info de env.step() avisa muchos frames mas tarde, 
        y yo quería agregar rewards negativos en el momento que muere)
        """
        obs_gris= self.rgb_to_grayscale2(obs)
        
        for kernel,value in zip(self.extra_kernel_list,self.extra_kernel_vals):
            convolved_image = convolve2d(obs_gris, kernel, mode='same')
            if np.any(convolved_image == value):
                return True
        return False

    def rgb_to_grayscale2(self,rgb_image):
        """
        Convierte a grayscale la imagen del juego.
        """
        grayscale_image = np.dot(rgb_image, [0.299, 0.587, 0.114])
        grayscale_image[(rgb_image.sum(axis=-1) > 0) == 0] = 0
        grayscale_image[grayscale_image > 0] = 255
        return grayscale_image.astype(np.uint8)
    
    def smart_flatten_short(self,list_of_matrices):
        """
        Recibe lista de matrices 7x(210,160) que contienen un valor '255' si en esa posicion cierto kernel detectó una forma específica (correspondiente a jugador,spider o centipede), y '0' si no.
        Este método se ocupa de convertir esa representación matricial en un array de largo 22 de las coordenadas, para que la use el modelo.

        También se ocupa de que si los kernels no encontraron al jugador, meter la última posición en la que se le vio. 
        Si no encontró ningun centipede meter la posición del medio de la cancha. Si no encontró un spider meterle la posición de un centipede.
        """
        
        flat_array=np.zeros(22,dtype=np.float32)
        
        player_x,player_y=np.where(list_of_matrices[0]==255)
        if len(player_x):
            flat_array[0]=player_x[0] 
            flat_array[1]=player_y[0] 
            self.past_player_pos[0]=player_x[0]
            self.past_player_pos[1]=player_y[0]
        else:
            #no hay player, le meto la ultima posición que tengo del player
            flat_array[0]=self.past_player_pos[0]
            flat_array[1]=self.past_player_pos[1]

        centipede_x,centipede_y=np.where(list_of_matrices[1]==255) 
        centipede_head_x,centipede_head_y=np.where(list_of_matrices[2]==255) 
        spider_x,spider_y=np.where(list_of_matrices[3]==255)
        if not len(spider_x):
            spider_x,spider_y=np.where(list_of_matrices[4]==255)
            if not len(spider_x):
                spider_x,spider_y=np.where(list_of_matrices[5]==255)
            if not len(spider_x):
                    spider_x,spider_y=np.where(list_of_matrices[6]==255)
            
        if len(spider_x):
            flat_array[20]=spider_x[0] 
            flat_array[21]=spider_y[0] 
        else:
            #no hubo spider en ningun kernel de spider, le meto posicion de centipede asi no queda (0,0)
            if len(centipede_x):
                flat_array[20]=centipede_x[0] 
                flat_array[21]=centipede_y[0] 
            elif len(centipede_head_x):
                #no hay centipedes, le meto un centipede head
                flat_array[20]=centipede_head_x[0] 
                flat_array[21]=centipede_head_y[0] 

        #ahora me ocupo de los centipedes que hayan quedado en zeros, (porque estan muertos), y si no hay ninguno, los pongo en media cancha hasta que aparezca alguno
        if len(centipede_x):
            flat_array[2:19:2]=np.tile(centipede_x,9)[:9] #intercaladamente mete los x, y los repite si no alcanzan, asi no quedan zeros. el tile es lo que repite, 9 veces por las dudas por si solo queda 1
            flat_array[3:20:2]=np.tile(centipede_y,9)[:9] #tapo los heads tambien de paso porlas dudas, casi nunca hay
        else:
            #no hay centipedes, meto centipede heads
            if len(centipede_head_x):
                flat_array[2:15:2]=np.tile(centipede_head_x,7)[:7]
                flat_array[3:16:2]=np.tile(centipede_head_y,7)[:7]
            else:
                #les meto el medio de la cancha pq es mejor que ir al borde
                flat_array[2:19:2]=np.tile([80],9)[:9]
                flat_array[3:20:2]=np.tile([70],9)[:9]

        #ahora los centipede head si hay, sino ya meti centipedes
        if len(centipede_head_x):
            flat_array[16:19:2]=np.tile(centipede_head_x,2)[:2] 
            flat_array[17:20:2]=np.tile(centipede_head_y,2)[:2]

        return flat_array