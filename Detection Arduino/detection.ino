
#include <Wire.h>
#include <VL53L0X.h>
#include <math.h>
VL53L0X sensor;

// step motor pin numbers
int IN1 = 10;
int IN2 = 11;
int IN3 = 12;
int IN4 = 13;
int led = 3;

//Variables for number of points estimation.
int detected_true=0,counter3=0;

// full-step mod stepper motor input arrays
int full_IN1[4] = {1,0,0,0};
int full_IN2[4] = {0,1,0,0};
int full_IN3[4] = {0,0,1,0};
int full_IN4[4] = {0,0,0,1};

float angle;
int k=0,counter5=0,x1,x2,y1,y2,counter4=0,control1=0,x=0;
int dir = 1; // initial direction, dir = 0 means clockwise rotation,  dir = 1 means counter-clockwise rotation
int counter = 0;
int step_number = 0;
float full_step_angle = 0.176,point;
//interrupt pin
int interrupt_pin=2;
unsigned long myTime,detection_time;
void setup() {
  Serial.begin(9600);
  
  Wire.begin();
  // timer interrupt settings
  cli();  // disable all interrupts
  TCCR1A = 0;   // reset control register A of TIMER1
  TCCR1B = 0;   // reset control register b of TIMER1
  TCNT1  = 0;   // actual value register of TIMER1
  OCR1A  = 124;   // output compare value 
  
  TCCR1B |= (1<<WGM12);   // enable the CTC mode
  TCCR1B |= (1<<CS12);    // prescale = 256
  TIMSK1 |= (1<<OCIE1A);  // enable the TIMER1 interrupts
  sei();  // enable all interrupts
  //attachInterrupt(digitalPinToInterrupt(interrupt_pin),makeAngleZero,LOW);
  // step motor pin assignments
  pinMode(IN1,OUTPUT);
  pinMode(IN2,OUTPUT);
  pinMode(IN3,OUTPUT);
  pinMode(IN4,OUTPUT);

 // sensor initialization settings
  pinMode(7,INPUT_PULLUP);
  pinMode(2,INPUT);
  digitalWrite(7 ,HIGH);
  sensor.init();
  sensor.setTimeout(500);
  //sensor.startContinuous();
  sensor.setMeasurementTimingBudget(20000); // High Speed mode = 20000 ns, Default Mode = 33000 ns, High Accuracy Mode = 200000 ns
}

void loop() { 
  int distance =sensor.readRangeSingleMillimeters();
  distance=closeDistanceElimination(distance);
  detected_true = agentDetection(distance);
  angle=angle_calculator();
  printPart(detected_true,distance);
  take_two_points(distance);
  detected_true=0;
  makeAngleZero();
  myTime=millis();
  Serial.print("mytime ");
  Serial.print(myTime);
  Serial.print("detection time ");
  Serial.print(detection_time);
 
  if (sensor.timeoutOccurred()) { Serial.print(" TIMEOUT"); }
  Serial.println();
}

ISR(TIMER1_COMPA_vect){
  digitalWrite(IN1,full_IN1[counter]);
  digitalWrite(IN2,full_IN2[counter]);
  digitalWrite(IN3,full_IN3[counter]);
  digitalWrite(IN4,full_IN4[counter]);
  
 if (dir == 0) {counter++; step_number--;}    // dir = 0 represents the clockwise rotation
 else if (dir == 1){counter--; step_number++;}  // dir = 1 represents the counter-clockwise rotation
  counter = counter_controller(dir, counter);
  if (step_number >= 2048){step_number=0;
  counter5++;}
 else if (step_number < 0){step_number=0;}
 if(counter4==1 && myTime>detection_time && control1==1)
 {
  dir=direction_controller(dir);
  control1=0;
 }
 else if(counter4==0 && myTime>detection_time && control1==2)
 {
  dir=direction_controller(dir);
  control1=0;
 }
}

int counter_controller(int dir,int counter){
  if (dir == 0 && counter > 3) {return 0;}
  else if (dir == 1 && counter < 0) {return 3;}
  else {return counter;}}
  
int direction_controller(int dir){
  if (dir == 0) {return 1;}
  else {return 0;}}

float angle_calculator(){
  angle = full_step_angle*step_number;
  return angle;
}

int closeDistanceElimination(int distance)
{
  if(distance<150){distance =150;}
  else if(distance>2000) {distance=2000;}
  return distance;
}
void take_two_points(int distance)
{
  if(counter3==2 && counter4==0)
  {
    x1=calculateXcoordinate(distance);
    y1=calculateYcoordinate(distance);
    counter4+=1;
    detection_time=myTime+1000;
    control1=1;
  }
  else if(counter3==2 && counter4==1)
  {
    x2=calculateXcoordinate(distance);
    y2=calculateYcoordinate(distance);
    counter4=0;
    detection_time=myTime+1000;
    control1=2;
  }
}
void makeAngleZero()
{
  if (digitalRead(2) == LOW && x==0) {
  step_number=0; 
  x++;}
  else if(digitalRead(2) == HIGH && x!=0)
  {
    x=0;
  }
}
int agentDetection(int distance)
{

    if (distance<1000) {detected_true = 1;
    counter3+=1;}
    else
    {
      detected_true = 0; 
      counter3 = 0;
     }
  return detected_true;
}
int calculateXcoordinate(int distance)
{
  int Xcoordinate;
  Xcoordinate=distance*cos(angle*0.01753);
  return Xcoordinate;
}
int calculateYcoordinate(int distance)
{
  int Ycoordinate;
  Ycoordinate=distance*sin(angle*0.01753);
  return Ycoordinate;
}
void printPart(int detected_true, int distance)
{
   if(detected_true==1)
  {
   Serial.print("Distance: ");
   Serial.print(distance);
   Serial.print("mm   ");
   digitalWrite(3,HIGH);
   Serial.print("angle: ");
   Serial.print(angle); 
   Serial.print(" Coordinates ");
   Serial.print(x1);
   Serial.print(" ");
   Serial.print(y1);
   Serial.print(" ");
   Serial.print(x2);
   Serial.print(" ");
   Serial.print(y2);
   Serial.print("   Detection: POSITIVE");
  }
 /* else 
  {
   Serial.print("Distance: ");
   Serial.print(distance);
   Serial.print("mm   ");
   digitalWrite(3,LOW);
   Serial.print("angle: ");
   Serial.print(angle); 
  }*/
}
