%% Amirreza Hatamipour TESTING PERCEPTION
% Bistable perception
% neuroscience project
%% Start
clear;
clc;
close all;
rng('shuffle');
sca % clear screen

%% Version 2

PsychDebugWindowConfiguration % add transparncy while in PTB
c=1;
Screen('Preference', 'SkipSyncTests', 1) ;
ScreenNumber = 0; 
[windowPtr, rect] = Screen('OpenWindow',ScreenNumber,0);
frameDuration = Screen('GetFlipInterval', windowPtr); % frame Duration

% 1. set up all the properties of the text we want to draw(font, size, style)
text_size = 48;
Screen('TextSize',windowPtr,text_size);

text_font = 'Arial';
Screen('TextFont',windowPtr,text_font);

text_style = 1; % 0:normal - 1:bold - 2:italic - 4:underline - 8:outline - 32:condence - 64:extend
Screen('TextStyle',windowPtr,text_style);


% 2. draw the text using DrawFormattedText
text = 'Hello Subject!\n Get ready for the experiment... \n please press a key';
color = [100 150 250];
DrawFormattedText(windowPtr,text,'center','center',color);

% show on screen
Screen('Flip',windowPtr);

%KbWait()
[secs_start, keyCode_start, deltaSecs_start] = KbWait();

pathToMovie = [pwd, '\Bistable_perception_DLBkwig3M2U_244.mkv'];


[moviePtr] = Screen('OpenMovie', windowPtr, pathToMovie);

Screen('PlayMovie', moviePtr, 1, 1, 0);

ResponseKey = zeros(1,400);
ResponseTime = zeros(1,400);
count=1;
%triger
%device = serial( 'COM8', 'BaudRate', 57600);
%fopen(device);
%fprintf( device, '100');
%pause(2)
%fprintf( device, '1');
%fclose(device);
t = GetSecs();
Start_time = t ;
toTime = Start_time + 85 ;
while t <  toTime 
      tex = Screen('GetMovieImage', windowPtr, moviePtr);
      if tex <= 0 
         break; 
      end
        
      Screen('DrawTexture', windowPtr, tex); 
      t = Screen('Flip', windowPtr);
      %[secs_action, keyCode_action, deltaSecs_action] = KbWait();
      [ keyIsDown, pressedSecs, keyCode] = KbCheck(0);
      if keyIsDown
         responseKey = KbName( find(keyCode ));
         ResponseKey(1,count) = responseKey; 
         responseTime = pressedSecs - Start_time;
         ResponseTime(1,count) = responseTime;
         count = count +1 ; 
      end
        Screen('Close', tex);
end
% show on screen
Screen('Flip',windowPtr);
s = 3; % in secs
WaitSecs(s)
Screen('PlayMovie', moviePtr, 0);
Screen('CloseMovie', moviePtr);
[moviePtr] = Screen('OpenMovie', windowPtr, pathToMovie);
Screen('PlayMovie', moviePtr, 1, 1, 0);
t = GetSecs();
Start_time_second = t - Start_time; 
toTime = t + 85 ;
while t <  toTime 
      tex = Screen('GetMovieImage', windowPtr, moviePtr);
      if tex <= 0 
         break; 
      end
        
      Screen('DrawTexture', windowPtr, tex); 
      t = Screen('Flip', windowPtr);
      %[secs_action, keyCode_action, deltaSecs_action] = KbWait();
      [ keyIsDown, pressedSecs, keyCode] = KbCheck(0);
      if keyIsDown
         responseKey = KbName( find(keyCode ));
         ResponseKey(1,count) = responseKey; 
         responseTime = pressedSecs - Start_time;
         ResponseTime(1,count) = responseTime;
         count = count +1 ; 
      end
        Screen('Close', tex);
end
 
 Screen('PlayMovie', moviePtr, 0);
 Screen('CloseMovie', moviePtr);
 clear Screen;
 %% save workspace
 c = c + 1 ; 
 file_name = ['data\Arshak_' , num2str(c)];
 save( file_name )
 
