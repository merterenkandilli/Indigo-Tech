
#include <Wire.h>
#include <VL53L0X.h>
#include <math.h>
VL53L0X sensor;

// step motor pin numbers
int IN1 = 13;
int IN2 = 12;
int IN3 = 11;
int IN4 = 10;

//dc motor pin numbers
int motor1pin1=5;
int motor1pin2=4;
int motor2pin1=3;
int motor2pin2=2;
int motor1control=9;
int motor2control=6;
int mspeed;
int radius;
//Variables for number of points estimation.
int detected_true=0,counter3=0;

// full-step mod stepper motor input arrays
int full_IN1[4] = {1,0,0,0};
int full_IN2[4] = {0,1,0,0};
int full_IN3[4] = {0,0,1,0};
int full_IN4[4] = {0,0,0,1};
int initial_arrange=1;
float angle,angle1,angle2;
int k=0,counter5=0,r1,r2=0,counter4=0,control1=0,x=0,stop_motion=0;
int dir = 1; // initial direction, dir = 0 means clockwise rotation,  dir = 1 means counter-clockwise rotation
int counter = 0;
int step_number = 0,step_number1,step_number2,speed_agent,rotate_time,forward_time;
float full_step_angle = 0.176,point;
//interrupt pin
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
  // dc motor initializations 
  pinMode(motor1pin1,OUTPUT);
  pinMode(motor1pin2,OUTPUT);
  pinMode(motor2pin1,OUTPUT);
  pinMode(motor2pin2,OUTPUT);
  pinMode(motor1control,OUTPUT);
  pinMode(motor2control,OUTPUT); 
 // sensor initialization settings
  pinMode(7,INPUT_PULLUP);
  pinMode(8,INPUT);
  digitalWrite(7 ,HIGH);
  sensor.init();
  sensor.setTimeout(500);
  //sensor.startContinuous();
  sensor.setMeasurementTimingBudget(20000); // High Speed mode = 20000 ns, Default Mode = 33000 ns, High Accuracy Mode = 200000 ns
  delay(3000);
  Serial.print("start");
}

void loop() { 
  if(initial_arrange==1)
  {
    if(digitalRead(8) == LOW)
    {
      step_number=0;
      initial_arrange=0;
    } 
  }
  else
  {
    int distance =sensor.readRangeSingleMillimeters();
    distance=closeDistanceElimination(distance);
    detected_true = agentDetection(distance);
 //   angle=angle_calculator();
    printPart(detected_true,distance);
    take_two_points(distance);
    detected_true=0;
    myTime=millis();
    calculateSpeed();
    if(r2!=0 && stop_motion==0)
    {
      if(angle2<90)
      {
        rotate_time = angle2*5;
        forward_time = distance*0.61;
        turn_around_left();
        delay(rotate_time);        
        stop_motor();
        delay(201);
        forward_straight();
        delay(forward_time);
        stop_motion=1;
        stop_motor();
      //  delay(rotate_time);
    //    stop_motor();
     //   delay(100);
   //     forward_straight();
    //    delay(forward_time);
   //     stop_motor();
      }
      else if(angle2<180 && angle2>90)
      {
        rotate_time = angle2*4;
        forward_time = distance*0.61;
        turn_around_left();
        delay(rotate_time);
        stop_motor();
        delay(201);
        forward_straight();
        delay(forward_time);
        stop_motion=1;
        stop_motor();
      }
      else if(angle2>180 && angle2<270)
      {
        rotate_time = ((360-angle2)*4.5);
        forward_time = distance*0.61;
        turn_around_right();
        delay(rotate_time);
        stop_motor();
        delay(201);
        forward_straight();
        delay(forward_time);
        stop_motion=1;
        stop_motor();
      }
      else if(angle2>270 && angle2<360)
      {
        rotate_time = ((360-angle2)*7);
        forward_time = distance*0.61;
        turn_around_right();
        delay(rotate_time);
        stop_motor();
        delay(201);
        forward_straight();
        delay(forward_time);
        stop_motion=1;
        stop_motor();
      }

    }/*
    Serial.print("mytime ");
    Serial.print(myTime);
    Serial.print("detection time ");
    Serial.print(detection_time);*/
 
    if (sensor.timeoutOccurred()) { Serial.print(" TIMEOUT"); }
    Serial.print(angle);
    Serial.println();
  }
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
 else if (step_number < 0){step_number=2047;}
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
 angle = full_step_angle*step_number;
}

int counter_controller(int dir,int counter){
  if (dir == 0 && counter > 3) {return 0;}
  else if (dir == 1 && counter < 0) {return 3;}
  else {return counter;}}
  
int direction_controller(int dir){
  if (dir == 0) {return 1;}
  else {return 0;}}
/*
float angle_calculator(){
  angle = full_step_angle*step_number;
  return angle;
}*/

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
    r1=distance;
    angle1=angle;
    step_number1=step_number;
    counter4+=1;
    detection_time=myTime+1000;
    control1=1;
  }
  else if(counter3==2 && counter4==1)
  {
    r2=distance;
    angle2=angle;
  //  Serial.print("angle   ");
  //  Serial.print(angle);
    counter4=0;
    step_number2=step_number;
    detection_time=myTime+1000;
    control1=2;
  }
}

/*
void makeAngleZero()
{
  if (digitalRead(8) == LOW && x==0) {
  step_number=0; 
  Serial.print("girdi");
  x++;}
  else if(digitalRead(8) == HIGH && x!=0)
  {
    x=0;
  }
}
*/

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

void calculateSpeed()
{
  if(r2!=0)
  {
    speed_agent=sqrt(square(r1)+square(r2)-2*(r1*r2*cos(angle2-angle1)))/(abs(step_number1-step_number2)*0.002);
  }
  else{speed_agent=0;}
}

void turn_around_right(){
  digitalWrite(motor1pin1, LOW);
  digitalWrite(motor1pin2, HIGH);
  analogWrite(motor1control, 255);

  digitalWrite(motor2pin1, HIGH);
  digitalWrite(motor2pin2, LOW);
  analogWrite(motor2control, 255);
}

void turn_around_left(){
  digitalWrite(motor1pin1, HIGH);
  digitalWrite(motor1pin2, LOW);
  analogWrite(motor1control, 255);

  digitalWrite(motor2pin1, LOW);
  digitalWrite(motor2pin2, HIGH);
  analogWrite(motor2control, 255);
}

void forward_straight(){
  digitalWrite(motor1pin1, HIGH);
  digitalWrite(motor1pin2, LOW);
  analogWrite(motor1control, 255);

  digitalWrite(motor2pin1, HIGH);
  digitalWrite(motor2pin2, LOW);
  analogWrite(motor2control, 255);
}

void backward_straight(){
  digitalWrite(motor1pin1, LOW);
  digitalWrite(motor1pin2, HIGH);
  analogWrite(motor1control, 255);

  digitalWrite(motor2pin1, LOW);
  digitalWrite(motor2pin2, HIGH);
  analogWrite(motor2control, 255);
}

void stop_motor(){
  digitalWrite(motor1pin1, LOW);
  digitalWrite(motor1pin2, HIGH);
  analogWrite(motor1control, 1);

  digitalWrite(motor2pin1, HIGH);
  digitalWrite(motor2pin2, LOW);
  analogWrite(motor2control, 1);
}

void printPart(int detected_true, int distance){
  if(detected_true==1)
  {
   Serial.print("Distance point 1: ");
   Serial.print(r1);
   Serial.print(" mm   ");
   Serial.print("Distance point 2: ");
   Serial.print(r2);
   Serial.print(" mm   ");
   Serial.print("angle1: ");
   Serial.print(angle1);
   Serial.print("angle2: ");
   Serial.print(angle2); 
 /*  Serial.print(" Coordinates ");
   Serial.print(x1);
   Serial.print(" ");
   Serial.print(y1);
   Serial.print(" ");
   Serial.print(x2);
   Serial.print(" ");
   Serial.print(y2);
   Serial.print("   Detection: POSITIVE");*/
  }
  else 
   {
 //  Serial.print("Distance: ");
 //  Serial.print(distance);
 //  Serial.print("mm   ");
 //  digitalWrite(3,LOW);
   Serial.print("Continues angle: ");
   Serial.print(angle); 
  }
}
