program project1;

{$mode objfpc}{$H+}

uses
  {$IFDEF UNIX}
  cthreads,
  {$ENDIF}
  {$IFDEF HASAMIGA}
  athreads,
  {$ENDIF}
  interfaces, // this includes the LCL widgetset
  forms, Unit1
  { you can add units after this };

{$R *.res}

begin
  requirederivedformresource:=true;
  application.scaled:=true;
  {$PUSH}{$WARN 5044 OFF}
  application.mainformontaskbar:=true;
  {$POP}
  application.initialize;
  application.createform(tformllamatest, formllamatest);
  application.run;
end.

