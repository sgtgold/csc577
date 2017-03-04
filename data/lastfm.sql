SELECT [UserID] AS UserID
	  ,[SongID]
	  ,[ListenTime]
  FROM [lastfm].[dbo].[playlist]
  WHERE UserID IN
  (
SELECT TOP 300 UserID
  FROM [lastfm].[dbo].[playlist]
  GROUP BY UserID
  ORDER BY COUNT(DISTINCT SongID) DESC
  )
  ORDER BY [UserID],[ListenTime] ASC

