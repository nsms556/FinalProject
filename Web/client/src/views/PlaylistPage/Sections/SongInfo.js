import React, {useEffect, useState} from "react";
import {Table} from "reactstrap";

import axios from "axios";

import Polarizing from "./Polarizing";


const SongInfo = () => {

    const [Songs, setSongs] = useState('');

    useEffect(() => {

        setSongs(test_data);

        // Songs 가져오기 -> 임시 데이터로 화면 구성
        // axios.get('http://127.0.0.1:8000/playlist/detail/')
        //     .then(response => {
        //         if (response.data) {
        //             setSongs(response.data.playlist_songs);
        //         } else {
        //             alert('Playlist(Songs)를 가져오지 못했습니다.');
        //         }
        //     })
    }, []);

    const test_data = [
        {
            "song_name": "안녕이라고 말하지마",
            "artist_name_basket": ["다비치"],
        },
        {
            "song_name": "참고 살아 (be...)",
            "artist_name_basket": ["다이나믹 듀오"],
        },
        {
            "song_name": "Vista",
            "artist_name_basket": ["피에스타"],
        },
        {
            "song_name": "Lollipop",
            "artist_name_basket": ["BIGBANG", "2NE1"],
        },
        {
            "song_name": "Fire",
            "artist_name_basket": ["2NE1"],
        },
    ];

    const renderSongs = Songs && Songs.map((song, _) => {
        return (
            <tr>
                <th scope="row">
                    <span className="mb-0 text-sm">
                        {song.song_name}
                    </span>
                </th>
                <td>
                    {song.artist_name_basket.join(", ")}
                </td>
                <td className="text-right">
                    <Polarizing/>
                </td>
            </tr>
        );
    });

    return (
        <Table className="align-items-center table-flush" responsive>
            <thead className="thead-light">
            <tr>
                <th scope="col">song_name</th>
                <th scope="col">artist_name_basket</th>
                <th scope="col"/>
            </tr>
            </thead>
            <tbody>
            {renderSongs}
            </tbody>
        </Table>
    );
};

export default SongInfo;
