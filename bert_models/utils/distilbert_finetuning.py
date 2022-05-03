import torch 
import torch.nn.functional as F 
from distilbert_utils import transfer_device,count_correct

def binary_cross_entropy(predictions, targets):
    loss =  -(targets * torch.log(predictions) + (1  - targets) * torch.log(1  - predictions))
    loss = torch.mean(loss)
    return loss

def  train_model(GPU, train_dataloader, dev_dataloader, tokenizer, model, optimizer, criterion,epochs):
    #Evaluate the performance of the model before training
    
    valid_loss, valid_accuracy = evaluate(GPU, dev_dataloader, model, criterion)
    print("Pre-training validation loss: "+str(valid_loss)+" --- Accuracy: "+str(valid_accuracy))
    print()

    #Train the model across 3 epochs and evaluate its performance
    for epoch in  range(epochs):
        model, train_loss, train_accuracy = train(GPU, train_dataloader, model, optimizer, criterion)
        valid_loss, valid_accuracy = evaluate(GPU, dev_dataloader, model, criterion)

        #Print performance stats
        print(" ", end="\r")
        print("Epoch: "+str(epoch+1))
        print("Training loss: "+str(train_loss)+" --- Accuracy: "+str(train_accuracy))
        print("Validation loss: "+str(valid_loss)+" --- Accuracy: "+str(valid_accuracy))
        print()
    return model


def  train(GPU, dataloader, model, optimizer, criterion):
    #Place the network in training mode, create a variable to store the total loss, and create a variable to store the total number of correct predictions
    model.train()
    total_loss =  0
    total_correct =  0

    #Loop through all batches in the dataloader
    for batch_number,(input_ids, input_mask_array, labels)  in  enumerate(dataloader):
        #Tokenize the text segments, get the model predictions, compute the loss, and add the loss to the total loss
        model_predictions = F.softmax(model(input_ids=transfer_device(GPU, input_ids), attention_mask=transfer_device(GPU, input_mask_array))['logits'], dim=1)

        loss = criterion(model_predictions, transfer_device(GPU, labels))
        
        total_loss += loss.item()

        #Count the number of correct predictions by the model in the batch and add this to the total correct
        correct = count_correct(model_predictions.cpu().detach().numpy(), labels.numpy())
        total_correct += correct

        #Zero the optimizer, compute the gradients, and update the model parameters
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print("Training batch index: "+str(batch_number)+"/"+str(len(dataloader))+  " ( "+str(batch_number/len(dataloader)*100)+"% )", end='\r')

    #Compute the average loss and accuracy across the epoch
    average_loss = total_loss /  len(dataloader)
    accuracy = total_correct / dataloader.dataset.__len__()
    return model, average_loss, accuracy


def  evaluate(GPU, dataloader, model, criterion):
    #Place the network in evaluation mode, create a variable to store the total loss, and create a variable to store the total number of correct predictions
    model.eval()
    total_loss =  0
    total_correct =  0

    #Loop through all batches in the dataloader
    for batch_number,(input_ids, input_mask_array, labels) in  enumerate(dataloader):
        #Tokenize the text segments, get the model predictions, compute the loss, and add the loss to the total loss

        model_predictions = F.softmax(model(input_ids=transfer_device(GPU, input_ids), attention_mask=transfer_device(GPU, input_mask_array))['logits'], dim=1)
        
        loss = criterion(model_predictions, transfer_device(GPU, labels))
        
        total_loss += loss.item()

        #Count the number of correct predictions by the model in the batch and add this to the total correct
        correct = count_correct(model_predictions.cpu().detach().numpy(), labels.numpy())
        total_correct += correct
        print("Evaluation batch index: "+str(batch_number)+"/"+str(len(dataloader))+  " ( "+str(batch_number/len(dataloader)*100)+"% )", end='\r')

    #Compute the average loss and accuracy across the epoch
    average_loss = total_loss /  len(dataloader)
    accuracy = total_correct / dataloader.dataset.__len__()
    return average_loss, accuracy



