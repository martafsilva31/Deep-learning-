import matplotlib.pyplot as plt
import numpy as np

def main():
    dataset = np.load('/Users/martasilva31/Library/CloudStorage/OneDrive-UniversidadedeLisboa/Mestrado/2 ano/P2/Aprendizagem profunda/Homeworks/Deep_learning_homeworks/Homework 1/skeleton_code/octmnist.npz')
    #print(dataset.keys())
    train_images = dataset['train_images']
    train_labels = dataset['train_labels']
    unique_labels = np.unique(train_labels)
    
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    for i, ax in enumerate(axes):
        label = unique_labels[i]
        label_indices = np.where(train_labels == label)[0]
        image_index = np.random.choice(label_indices)
        
        image = train_images[image_index].reshape((28, 28))
        ax.imshow(image, cmap='gray')
        ax.set_title(f"Label {int(label)}")
        #ax.axis('off')
    
    #plt.tight_layout()
    plt.savefig('/Users/martasilva31/Library/CloudStorage/OneDrive-UniversidadedeLisboa/Mestrado/2 ano/P2/Aprendizagem profunda/Homeworks/Deep_learning_homeworks/Homework 1/octmnist.jpeg')
    
if __name__ == "__main__":
    main()
